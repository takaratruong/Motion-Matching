#include "interaction_runtime.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace interaction {
namespace {

constexpr float kCanonicalFps = 25.0F;
constexpr float kFixedDt = 1.0F / kCanonicalFps;
constexpr float kMinimumPlaybackSpeed = 0.85F;
constexpr float kMaximumPlaybackSpeed = 1.15F;
constexpr float kSlabEpsilon = 1.0e-7F;
constexpr double kEventTimeEpsilon = 1.0e-6;
constexpr double kEventSourceTolerance = 1.0e-4;

bool finite_nonnegative(float value) {
    return std::isfinite(value) && value >= 0.0F;
}

bool valid_match_config(const MatchConfig& config) {
    if (!finite_nonnegative(config.maximum_approach_m) ||
        !finite_nonnegative(config.maximum_root_correction_m) ||
        !finite_nonnegative(config.maximum_yaw_correction_radians) ||
        !finite_nonnegative(config.maximum_hand_correction_m) ||
        !finite_nonnegative(config.maximum_hand_orientation_radians) ||
        !finite_nonnegative(config.maximum_cost)) {
        return false;
    }
    float weight_sum = 0.0F;
    for (float weight : config.group_weights) {
        if (!finite_nonnegative(weight)) return false;
        weight_sum += weight;
    }
    return std::isfinite(weight_sum) && weight_sum > 0.0F;
}

bool valid_playback_config(const PlaybackConfig& config) {
    return config.canonical_fps == kCanonicalFps &&
           std::isfinite(config.speed) &&
           std::isfinite(config.minimum_speed) &&
           std::isfinite(config.maximum_speed) &&
           config.minimum_speed >= kMinimumPlaybackSpeed &&
           config.maximum_speed <= kMaximumPlaybackSpeed &&
           config.minimum_speed <= config.maximum_speed &&
           config.speed >= config.minimum_speed &&
           config.speed <= config.maximum_speed &&
           finite_nonnegative(config.entry_blend_seconds) &&
           finite_nonnegative(config.maximum_alignment_seconds) &&
           finite_nonnegative(config.commit_horizon_seconds);
}

bool valid_ik_config(const IKConfig& config) {
    return finite_nonnegative(config.maximum_request_position_m) &&
           finite_nonnegative(config.maximum_request_orientation_radians) &&
           finite_nonnegative(config.accepted_position_m) &&
           finite_nonnegative(config.accepted_orientation_radians) &&
           std::isfinite(config.damping) && config.damping > 0.0F &&
           std::isfinite(config.finite_difference_radians) &&
           config.finite_difference_radians > 0.0F &&
           std::isfinite(config.orientation_scale_m_per_radian) &&
           config.orientation_scale_m_per_radian > 0.0F &&
           std::isfinite(config.maximum_step_radians) &&
           config.maximum_step_radians > 0.0F &&
           config.maximum_iterations >= 0;
}

bool valid_attachment_config(const AttachmentConfig& config) {
    return finite_nonnegative(config.maximum_position_error_m) &&
           finite_nonnegative(config.maximum_orientation_error_radians) &&
           finite_nonnegative(config.required_lift_m) &&
           finite_nonnegative(config.required_hold_seconds);
}

bool valid_carry_config(const CarryConfig& config) {
    return config.minimum_hold_frames >= 2 &&
           finite_nonnegative(config.maximum_grasp_drift_m) &&
           finite_nonnegative(config.maximum_grasp_drift_radians) &&
           finite_nonnegative(config.minimum_root_displacement_m) &&
           finite_nonnegative(config.minimum_average_speed_mps) &&
           std::isfinite(config.search_interval_seconds) &&
           config.search_interval_seconds > 0.0F &&
           std::isfinite(config.spine_weight) &&
           config.spine_weight >= 0.0F && config.spine_weight <= 1.0F &&
           std::isfinite(config.inactive_arm_weight) &&
           config.inactive_arm_weight >= 0.0F &&
           config.inactive_arm_weight <= 1.0F;
}

void validate_runtime_config(const RuntimeConfig& config) {
    if (!valid_match_config(config.matcher) ||
        !valid_playback_config(config.playback) ||
        !valid_ik_config(config.ik) ||
        !valid_attachment_config(config.attachment) ||
        !valid_carry_config(config.carry)) {
        throw std::invalid_argument("invalid interaction runtime config");
    }
}

bool segment_intersects_expanded_box(
    vec3 start_world,
    vec3 stop_world,
    Transform box,
    vec3 box_size,
    float expansion) {
    const quat box_from_world = quat_inv(box.rotation);
    const vec3 start = quat_mul_vec3(
        box_from_world, start_world - box.position);
    const vec3 stop = quat_mul_vec3(
        box_from_world, stop_world - box.position);
    const vec3 delta = stop - start;
    const vec3 half = 0.5F * box_size + expansion;
    const std::array<float, 3> starts = {start.x, start.y, start.z};
    const std::array<float, 3> deltas = {delta.x, delta.y, delta.z};
    const std::array<float, 3> halves = {half.x, half.y, half.z};
    float minimum = 0.0F;
    float maximum = 1.0F;
    for (size_t axis = 0; axis < starts.size(); ++axis) {
        if (std::abs(deltas[axis]) <= kSlabEpsilon) {
            if (starts[axis] < -halves[axis] ||
                starts[axis] > halves[axis]) {
                return false;
            }
            continue;
        }
        float first = (-halves[axis] - starts[axis]) / deltas[axis];
        float second = (halves[axis] - starts[axis]) / deltas[axis];
        if (first > second) std::swap(first, second);
        minimum = std::max(minimum, first);
        maximum = std::min(maximum, second);
        if (minimum > maximum) return false;
    }
    return true;
}

RuntimeOutput passthrough(
    const RuntimeInput& input,
    const RuntimeDiagnostics& diagnostics) {
    RuntimeOutput output{};
    output.pose = input.locomotion.pose;
    output.diagnostics = diagnostics;
    return output;
}

QueryInput make_query_input(
    const LocomotionSnapshot& locomotion,
    const InteractionTarget& target,
    const GraspAffordance& affordance) {
    QueryInput query{};
    query.locomotion = locomotion;
    query.grasp_world = compose(target.object_world, affordance.hand_in_object);
    query.table_world = target.table_world;
    query.table_size = target.table_size;
    query.approach_direction_object =
        affordance.approach_direction_object;
    query.object_dimensions = target.object_dimensions;
    query.hand = affordance.hand;
    return query;
}

bool exact(vec3 left, vec3 right) {
    return left.x == right.x && left.y == right.y && left.z == right.z;
}

bool exact(quat left, quat right) {
    return left.w == right.w && left.x == right.x &&
           left.y == right.y && left.z == right.z;
}

bool exact(const Transform& left, const Transform& right) {
    return exact(left.position, right.position) &&
           exact(left.rotation, right.rotation);
}

bool exact(
    const GraspAffordance& left,
    const GraspAffordance& right) {
    return left.id == right.id && left.hand == right.hand &&
           exact(left.hand_in_object, right.hand_in_object) &&
           exact(
               left.approach_direction_object,
               right.approach_direction_object) &&
           left.clearance_radius == right.clearance_radius;
}

bool exact(
    const InteractionTarget& left,
    const InteractionTarget& right) {
    if (left.handle != right.handle ||
        !exact(left.object_world, right.object_world) ||
        !exact(left.object_dimensions, right.object_dimensions) ||
        !exact(left.table_world, right.table_world) ||
        !exact(left.table_size, right.table_size) ||
        left.state != right.state ||
        left.owner_request != right.owner_request ||
        left.affordances.size() != right.affordances.size()) {
        return false;
    }
    for (size_t index = 0; index < left.affordances.size(); ++index) {
        if (!exact(left.affordances[index], right.affordances[index])) {
            return false;
        }
    }
    return true;
}

Transform hand_world(const Pose& pose, Hand hand) {
    const WorldPose world = world_pose(pose);
    const size_t bone = hand == Hand::Left
        ? kLeftHandBone
        : kRightHandBone;
    return {world.positions[bone], world.rotations[bone]};
}

IKResult apply_reach_ik(
    Pose& pose,
    int32_t frame,
    const MatchCandidate& candidate,
    const GraspAffordance& affordance,
    Transform target_hand_world,
    Transform source_contact_hand_world,
    const IKConfig& config) {
    const Transform sampled_hand = hand_world(pose, affordance.hand);
    const float reach_alpha = std::clamp(
        static_cast<float>(frame - candidate.entry_frame) /
            static_cast<float>(
                candidate.contact_frame - candidate.entry_frame),
        0.0F,
        1.0F);
    Transform corrected_hand{};
    corrected_hand.position = sampled_hand.position + reach_alpha * (
        target_hand_world.position - source_contact_hand_world.position);
    const quat full_rotation = quat_normalize(quat_mul(
        quat_mul(
            target_hand_world.rotation,
            quat_inv(source_contact_hand_world.rotation)),
        sampled_hand.rotation));
    corrected_hand.rotation = quat_nlerp_shortest(
        sampled_hand.rotation, full_rotation, reach_alpha);
    return solve_hand_ik(
        pose, affordance.hand, corrected_hand, config);
}

Pose apply_entry_blend(
    const Pose& source,
    const Pose& sampled,
    float elapsed_seconds,
    float blend_seconds) {
    const float alpha = blend_seconds <= 0.0F
        ? 1.0F
        : std::clamp(elapsed_seconds / blend_seconds, 0.0F, 1.0F);
    return interpolate_pose(source, sampled, alpha);
}

struct EventPose {
    Pose pose{};
    bool joints_valid = true;
    float hand_position_error_m = 0.0F;
    float hand_orientation_error_radians = 0.0F;
};

EventPose event_pose(
    const SequentialPlayer& player,
    const MatchCandidate& candidate,
    const GraspAffordance& affordance,
    Transform target_hand_world,
    Transform source_contact_hand_world,
    const IKConfig& ik_config,
    const Pose& entry_blend_source,
    float entry_blend_seconds) {
    EventPose result{};
    result.pose = player.sample();
    if (player.frame() <= candidate.contact_frame) {
        const IKResult ik = apply_reach_ik(
            result.pose,
            player.frame(),
            candidate,
            affordance,
            target_hand_world,
            source_contact_hand_world,
            ik_config);
        result.joints_valid = ik.accepted;
        result.hand_position_error_m = ik.position_error_m;
        result.hand_orientation_error_radians =
            ik.orientation_error_radians;
    }
    result.pose = apply_entry_blend(
        entry_blend_source,
        result.pose,
        player.elapsed_seconds(),
        entry_blend_seconds);
    return result;
}

double source_frame_at_elapsed(
    const Database& database,
    const MatchCandidate& candidate,
    float elapsed_seconds,
    float speed) {
    const int32_t final_frame = database.range_stops.at(
        static_cast<size_t>(candidate.clip)) - 1;
    const double source = static_cast<double>(candidate.entry_frame) +
        static_cast<double>(elapsed_seconds) *
            static_cast<double>(kCanonicalFps) *
            static_cast<double>(speed);
    const double clamped = std::min(
        source, static_cast<double>(final_frame));
    const double nearest = std::round(clamped);
    return std::abs(clamped - nearest) <= kEventSourceTolerance
        ? nearest
        : clamped;
}

float credited_above_threshold_seconds(
    float segment_seconds,
    float start_height,
    float stop_height,
    float threshold) {
    if (stop_height < threshold) return segment_seconds;
    if (start_height >= threshold) return segment_seconds;
    if (!(stop_height > start_height)) return 0.0F;
    const float crossing_alpha = std::clamp(
        (threshold - start_height) / (stop_height - start_height),
        0.0F,
        1.0F);
    return segment_seconds * (1.0F - crossing_alpha);
}

void accumulate_runtime_clearance(
    bool& path_clear,
    bool& has_previous,
    vec3& previous_hand_world,
    vec3 current_hand_world,
    const InteractionTarget& target,
    const GraspAffordance& affordance,
    bool exempt_target_object) {
    const vec3 start = has_previous
        ? previous_hand_world
        : current_hand_world;
    if (path_clear && segment_intersects_expanded_box(
            start,
            current_hand_world,
            target.table_world,
            target.table_size,
            affordance.clearance_radius)) {
        path_clear = false;
    }
    if (path_clear && !exempt_target_object &&
        segment_intersects_expanded_box(
            start,
            current_hand_world,
            target.object_world,
            target.object_dimensions,
            affordance.clearance_radius)) {
        path_clear = false;
    }
    previous_hand_world = current_hand_world;
    has_previous = true;
}

Pose mapped_pose_at_frame(
    const Database& database,
    const MatchCandidate& candidate,
    int32_t frame) {
    Pose pose = pose_at_frame(database, frame);
    const size_t root = g1_skeleton::Simulation;
    const Transform mapped_root = compose(
        candidate.scene_from_source,
        Transform{pose.positions[root], pose.rotations[root]});
    pose.positions[root] = mapped_root.position;
    pose.rotations[root] = mapped_root.rotation;
    return pose;
}

float correction_weight(
    const MatchCandidate& candidate,
    const SequentialPlayer& player,
    float speed) {
    const double source = static_cast<double>(candidate.entry_frame) +
        static_cast<double>(player.elapsed_seconds()) * kCanonicalFps *
            static_cast<double>(speed);
    if (source <= candidate.entry_frame) return 1.0F;
    if (source >= candidate.contact_frame) return 0.0F;
    const double alpha =
        (source - static_cast<double>(candidate.entry_frame)) /
        static_cast<double>(
            candidate.contact_frame - candidate.entry_frame);
    const double smoothstep = alpha * alpha * (3.0 - 2.0 * alpha);
    return static_cast<float>(1.0 - smoothstep);
}

ContactMeasurement contact_measurement(
    const Database& database,
    const Pose& pose,
    int32_t frame,
    TargetHandle target,
    Hand hand,
    Transform target_hand,
    bool stable_contact_event,
    bool joints_valid,
    bool clearance_valid) {
    const Transform measured_hand = hand_world(pose, hand);
    const size_t hand_index = hand == Hand::Left ? 0U : 1U;
    ContactMeasurement measurement{};
    measurement.target = target;
    measurement.hand = hand;
    measurement.hand_world = measured_hand;
    measurement.position_error_m = length(
        measured_hand.position - target_hand.position);
    measurement.orientation_error_radians = quat_angle_between(
        measured_hand.rotation, target_hand.rotation);
    measurement.stable_contact_event = stable_contact_event;
    measurement.hand_contact = database.hand_contacts.at(
        static_cast<size_t>(frame) * 2U + hand_index) != 0U;
    measurement.joints_valid = joints_valid;
    measurement.clearance_valid = clearance_valid;
    return measurement;
}

}  // namespace

InteractionRuntime::InteractionRuntime(
    const Database& database,
    const Features& features,
    TargetRegistry& registry,
    RuntimeConfig config)
    : database_(&database),
      features_(&features),
      registry_(&registry),
      config_(config),
      state_(RuntimeState::Locomotion) {
    validate_runtime_config(config_);
    diagnostics_.state = state_;
    diagnostics_.playback_speed = config_.playback.speed;
    diagnostics_.pack_available = true;
}

InteractionRuntime InteractionRuntime::disabled(Reason reason) {
    InteractionRuntime runtime;
    runtime.state_ = RuntimeState::Disabled;
    runtime.diagnostics_.state = RuntimeState::Disabled;
    runtime.diagnostics_.reason = reason;
    runtime.diagnostics_.pack_available = false;
    return runtime;
}

RuntimeState InteractionRuntime::state() const {
    return state_;
}

const RuntimeDiagnostics& InteractionRuntime::diagnostics() const {
    return diagnostics_;
}

void InteractionRuntime::drain_playback_events(
    float published_elapsed_seconds) {
    if (database_ == nullptr || !candidate_.has_value() ||
        !affordance_.has_value() || !target_.has_value() ||
        !request_.has_value() || !player_.has_value() ||
        !event_player_.has_value() || !attachment_.has_value()) {
        throw std::logic_error("interaction runtime event state is incomplete");
    }

    const float target_elapsed_seconds = player_->elapsed_seconds();
    const float source_rate = kCanonicalFps * config_.playback.speed;
    const int32_t final_frame = database_->range_stops.at(
        static_cast<size_t>(candidate_->clip)) - 1;
    while (static_cast<double>(target_elapsed_seconds) -
               static_cast<double>(event_player_->elapsed_seconds()) >
           kEventTimeEpsilon) {
        const float event_elapsed_seconds =
            event_player_->elapsed_seconds();
        const double start_source = source_frame_at_elapsed(
            *database_,
            *candidate_,
            event_elapsed_seconds,
            config_.playback.speed);
        const float remaining_seconds = std::max(
            0.0F, target_elapsed_seconds - event_elapsed_seconds);
        float segment_seconds = remaining_seconds;
        if (start_source <
            static_cast<double>(final_frame) - kEventTimeEpsilon) {
            const double next_source =
                std::floor(start_source + kEventTimeEpsilon) + 1.0;
            const double to_boundary_seconds =
                (next_source - start_source) /
                static_cast<double>(source_rate);
            if (to_boundary_seconds > kEventTimeEpsilon) {
                segment_seconds = std::min(
                    segment_seconds,
                    static_cast<float>(to_boundary_seconds));
            }
        }
        if (!(segment_seconds > 0.0F)) break;

        const EventPose start = event_pose(
            *event_player_,
            *candidate_,
            *affordance_,
            target_hand_world_,
            source_contact_hand_world_,
            config_.ik,
            entry_blend_source_,
            config_.playback.entry_blend_seconds);
        const bool attached_before_segment =
            ever_attached_ && !post_commit_failure_;
        const bool carry_ready_before_segment = carry_ready_;
        const float start_object_height = compose(
            hand_world(start.pose, affordance_->hand),
            inverse(affordance_->hand_in_object)).position.y;

        event_player_->advance(segment_seconds);
        const EventPose stop = event_pose(
            *event_player_,
            *candidate_,
            *affordance_,
            target_hand_world_,
            source_contact_hand_world_,
            config_.ik,
            entry_blend_source_,
            config_.playback.entry_blend_seconds);
        const double stop_source = source_frame_at_elapsed(
            *database_,
            *candidate_,
            event_player_->elapsed_seconds(),
            config_.playback.speed);
        diagnostics_.hand_position_error_m =
            stop.hand_position_error_m;
        diagnostics_.hand_orientation_error_radians =
            stop.hand_orientation_error_radians;

        if (stop_source <=
            static_cast<double>(candidate_->contact_frame) +
                kEventTimeEpsilon) {
            const bool exempt_target_object = stop_source >
                static_cast<double>(candidate_->contact_frame - 1) +
                    kEventTimeEpsilon;
            accumulate_runtime_clearance(
                corrected_path_clear_,
                has_previous_corrected_hand_,
                previous_corrected_hand_world_,
                hand_world(stop.pose, affordance_->hand).position,
                *target_,
                *affordance_,
                exempt_target_object);
        }

        if (!post_commit_failure_ && !contact_evaluated_ &&
            event_player_->frame() >= candidate_->contact_frame) {
            const ContactMeasurement contact = contact_measurement(
                *database_,
                stop.pose,
                candidate_->contact_frame,
                request_->target,
                affordance_->hand,
                target_hand_world_,
                true,
                stop.joints_valid,
                corrected_path_clear_);
            diagnostics_.hand_position_error_m =
                contact.position_error_m;
            diagnostics_.hand_orientation_error_radians =
                contact.orientation_error_radians;
            contact_evaluated_ = true;
            if (!attachment_->try_contact(contact)) {
                post_commit_failure_ = true;
                post_commit_reason_ = attachment_->reason();
                diagnostics_.result = ResultCode::Failed;
                diagnostics_.reason = post_commit_reason_;
                diagnostics_.attached = false;
            } else {
                ever_attached_ = true;
                object_world_ = attachment_->object_world();
                diagnostics_.object_state = ObjectState::Attached;
                diagnostics_.attached = true;
            }
        }

        if (attached_before_segment && !post_commit_failure_) {
            const int32_t start_frame = static_cast<int32_t>(
                std::floor(start_source + kEventTimeEpsilon));
            const ContactMeasurement measurement = contact_measurement(
                *database_,
                stop.pose,
                event_player_->frame(),
                request_->target,
                affordance_->hand,
                target_hand_world_,
                false,
                stop.joints_valid,
                true);
            diagnostics_.hand_position_error_m =
                measurement.position_error_m;
            diagnostics_.hand_orientation_error_radians =
                measurement.orientation_error_radians;
            const float stop_object_height = compose(
                measurement.hand_world,
                inverse(affordance_->hand_in_object)).position.y;
            const float lift_threshold =
                original_object_world_.position.y +
                config_.attachment.required_lift_m;
            auto apply_attachment_update = [this](
                                               const ContactMeasurement& value,
                                               float seconds) {
                attachment_->update(value, seconds);
                object_world_ = attachment_->object_world();
                if (attachment_->result() == ResultCode::Failed) {
                    post_commit_failure_ = true;
                    post_commit_reason_ = attachment_->reason();
                    diagnostics_.result = ResultCode::Failed;
                    diagnostics_.reason = post_commit_reason_;
                    diagnostics_.attached = false;
                    return false;
                }
                diagnostics_.object_state = attachment_->state();
                diagnostics_.attached = true;
                return true;
            };

            if (carry_ready_before_segment) {
                ContactMeasurement held_measurement = measurement;
                held_measurement.hand_contact = true;
                (void)apply_attachment_update(held_measurement, 0.0F);
                continue;
            }

            ContactMeasurement interval_measurement = measurement;
            const size_t hand_index = affordance_->hand == Hand::Left
                ? 0U
                : 1U;
            interval_measurement.hand_contact =
                database_->hand_contacts.at(
                    static_cast<size_t>(start_frame) * 2U + hand_index) != 0U;
            bool continuous_update_succeeded = true;
            if (start_object_height >= lift_threshold &&
                stop_object_height < lift_threshold) {
                const float crossing_alpha = std::clamp(
                    (start_object_height - lift_threshold) /
                        (start_object_height - stop_object_height),
                    0.0F,
                    1.0F);
                ContactMeasurement crossing = contact_measurement(
                    *database_,
                    start.pose,
                    start_frame,
                    request_->target,
                    affordance_->hand,
                    target_hand_world_,
                    false,
                    start.joints_valid,
                    true);
                crossing.hand_world.position =
                    crossing.hand_world.position + crossing_alpha *
                        (measurement.hand_world.position -
                         crossing.hand_world.position);
                crossing.hand_world.rotation = quat_nlerp_shortest(
                    crossing.hand_world.rotation,
                    measurement.hand_world.rotation,
                    crossing_alpha);
                const Transform crossing_object = compose(
                    crossing.hand_world,
                    inverse(affordance_->hand_in_object));
                crossing.hand_world.position.y +=
                    lift_threshold - crossing_object.position.y;
                const float above_seconds =
                    segment_seconds * crossing_alpha;
                if (above_seconds > 0.0F &&
                    !apply_attachment_update(crossing, above_seconds)) {
                    continuous_update_succeeded = false;
                }
                if (continuous_update_succeeded) {
                    continuous_update_succeeded = apply_attachment_update(
                        interval_measurement,
                        segment_seconds - above_seconds);
                }
            } else {
                const float attachment_seconds =
                    credited_above_threshold_seconds(
                        segment_seconds,
                        start_object_height,
                        stop_object_height,
                        lift_threshold);
                continuous_update_succeeded = apply_attachment_update(
                    interval_measurement, attachment_seconds);
            }
            if (!continuous_update_succeeded) continue;

            if (attachment_->state() == ObjectState::Held &&
                stop_source + kEventSourceTolerance >=
                    static_cast<double>(candidate_->hold_frame)) {
                carry_ready_ = true;
                float terminal_elapsed_seconds =
                    event_player_->elapsed_seconds();
                Pose terminal_pose = stop.pose;
                int32_t terminal_frame = event_player_->frame();
                Phase terminal_phase = event_player_->phase();
                if (start_source <
                        static_cast<double>(candidate_->hold_frame) &&
                    stop_source + kEventSourceTolerance >=
                        static_cast<double>(candidate_->hold_frame)) {
                    terminal_elapsed_seconds =
                        static_cast<float>(
                            candidate_->hold_frame -
                            candidate_->entry_frame) /
                        source_rate;
                    terminal_pose = mapped_pose_at_frame(
                        *database_, *candidate_, candidate_->hold_frame);
                    terminal_pose = apply_entry_blend(
                        entry_blend_source_,
                        terminal_pose,
                        terminal_elapsed_seconds,
                        config_.playback.entry_blend_seconds);
                    terminal_frame = candidate_->hold_frame;
                    terminal_phase = Phase::Hold;
                    ContactMeasurement terminal_measurement =
                        contact_measurement(
                            *database_,
                            terminal_pose,
                            terminal_frame,
                            request_->target,
                            affordance_->hand,
                            target_hand_world_,
                            false,
                            true,
                            true);
                    terminal_measurement.hand_contact = true;
                    if (!apply_attachment_update(
                            terminal_measurement, 0.0F)) {
                        continue;
                    }
                }
                if (terminal_elapsed_seconds +
                        static_cast<float>(kEventTimeEpsilon) >=
                    published_elapsed_seconds) {
                    pose_ = terminal_pose;
                    diagnostics_.frame = terminal_frame;
                    diagnostics_.phase = terminal_phase;
                    diagnostics_.applied_root_correction_m = length(
                        event_player_->entry_root_correction());
                    diagnostics_.applied_yaw_correction_radians =
                        diagnostics_.requested_yaw_correction_radians *
                        correction_weight(
                            *candidate_,
                            *event_player_,
                            config_.playback.speed);
                    return;
                }
            }
            if (!carry_ready_) {
                (void)apply_attachment_update(measurement, 0.0F);
            }
        }
    }

    if (carry_ready_ && !post_commit_failure_) {
        ContactMeasurement held_measurement = contact_measurement(
            *database_,
            pose_,
            player_->frame(),
            request_->target,
            affordance_->hand,
            target_hand_world_,
            false,
            true,
            true);
        held_measurement.hand_contact = true;
        attachment_->update(held_measurement, 0.0F);
        object_world_ = attachment_->object_world();
        diagnostics_.hand_position_error_m =
            held_measurement.position_error_m;
        diagnostics_.hand_orientation_error_radians =
            held_measurement.orientation_error_radians;
        if (attachment_->result() == ResultCode::Failed) {
            carry_ready_ = false;
            post_commit_failure_ = true;
            post_commit_reason_ = attachment_->reason();
            diagnostics_.result = ResultCode::Failed;
            diagnostics_.reason = post_commit_reason_;
            diagnostics_.attached = false;
        }
    }
}

void InteractionRuntime::begin_carry() {
    if (!candidate_.has_value() || !affordance_.has_value() ||
        !attachment_.has_value() || !carry_ready_) {
        throw std::logic_error("interaction runtime carry state is incomplete");
    }
    if (!carry_started_) {
        carry_.emplace(
            *database_,
            *features_,
            classify_carry_ranges(*database_, config_.carry),
            config_.carry,
            config_.ik);
        carry_->start(
            pose_, affordance_->hand, *affordance_, object_world_);
        carry_started_ = true;
    }
    state_ = RuntimeState::Carry;
    diagnostics_.state = state_;
    diagnostics_.result = ResultCode::Succeeded;
    diagnostics_.reason = Reason::None;
    diagnostics_.object_state = ObjectState::Held;
    diagnostics_.attached = true;
    diagnostics_.recorded_carry = false;
}

RuntimeOutput InteractionRuntime::update(const RuntimeInput& input) {
    if (state_ == RuntimeState::Disabled) {
        return passthrough(input, diagnostics_);
    }
    if (input.dt != kFixedDt) {
        throw std::invalid_argument(
            "interaction runtime dt must be exactly 1/25 second");
    }

    if (state_ == RuntimeState::Locomotion && input.interact_pressed) {
        request_ = input.pick_request;
        owns_reservation_ = false;
        diagnostics_ = RuntimeDiagnostics{};
        diagnostics_.state = RuntimeState::Preflight;
        diagnostics_.playback_speed = config_.playback.speed;
        diagnostics_.pack_available = true;
        if (request_.has_value()) {
            diagnostics_.target = request_->target;
            diagnostics_.affordance_id = request_->affordance_id;
        }
        state_ = RuntimeState::Preflight;
    } else if (state_ == RuntimeState::Preflight && !request_.has_value()) {
        state_ = RuntimeState::Locomotion;
        diagnostics_.state = state_;
        diagnostics_.result = ResultCode::Rejected;
        diagnostics_.reason = Reason::TargetUnavailable;
    } else if (state_ == RuntimeState::Preflight) {
        const PickRequest request = *request_;
        auto reject = [this](Reason reason) {
            if (owns_reservation_ && request_.has_value() &&
                registry_ != nullptr) {
                (void)registry_->release(
                    request_->target, request_->request_id);
            }
            owns_reservation_ = false;
            state_ = RuntimeState::Locomotion;
            diagnostics_.state = state_;
            diagnostics_.result = ResultCode::Rejected;
            diagnostics_.reason = reason;
            diagnostics_.object_state = ObjectState::Free;
            diagnostics_.attached = false;
        };

        if (request.target.id == 0U || request.target.generation == 0U ||
            request.request_id == 0U) {
            reject(Reason::TargetUnavailable);
        } else {
            const InteractionTarget* available = registry_->find(
                request.target);
            if (available == nullptr) {
                reject(Reason::TargetChanged);
            } else {
                const GraspAffordance* requested_affordance =
                    registry_->find_affordance(
                        request.target, request.affordance_id);
                if (requested_affordance == nullptr) {
                    reject(Reason::TargetUnavailable);
                } else {
                    original_object_world_ = available->object_world;
                    object_world_ = original_object_world_;
                    if (!registry_->reserve(
                            request.target, request.request_id)) {
                        reject(Reason::TargetChanged);
                    } else {
                        owns_reservation_ = true;
                        try {
                            const InteractionTarget* reserved = registry_->find(
                                request.target);
                            const GraspAffordance* reserved_affordance =
                                registry_->find_affordance(
                                    request.target, request.affordance_id);
                            if (reserved == nullptr ||
                                reserved_affordance == nullptr ||
                                !registry_->validate(
                                    request.target, request.request_id)) {
                                reject(Reason::TargetChanged);
                            } else {
                                target_ = *reserved;
                                affordance_ = *reserved_affordance;
                                const NormalizedQuery query = normalize_query(
                                    build_raw_query(make_query_input(
                                        input.locomotion,
                                        *target_,
                                        *affordance_)),
                                    *features_);
                                MatchInput match_input{};
                                match_input.database = database_;
                                match_input.features = features_;
                                match_input.query = query;
                                match_input.locomotion = input.locomotion;
                                match_input.target = *target_;
                                match_input.affordance = *affordance_;
                                match_input.request = request;
                                const MatchResult match = select_whole_clip(
                                    match_input, config_.matcher);
                                if (!match.accepted) {
                                    reject(match.reason);
                                } else {
                                    candidate_ = match.candidate;
                                    player_.emplace(*database_);
                                    player_->start(
                                        *candidate_,
                                        input.locomotion.pose,
                                        config_.playback.speed);
                                    event_player_.emplace(*database_);
                                    event_player_->start(
                                        *candidate_,
                                        input.locomotion.pose,
                                        config_.playback.speed);
                                    attachment_.emplace(
                                        *registry_, config_.attachment);
                                    if (!attachment_->begin(
                                            *target_,
                                            request,
                                            *affordance_,
                                            original_object_world_.position.y)) {
                                        reject(attachment_->reason());
                                    } else {
                                        entry_blend_source_ =
                                            input.locomotion.pose;
                                        pose_ = entry_blend_source_;
                                        target_hand_world_ = compose(
                                            target_->object_world,
                                            affordance_->hand_in_object);
                                        source_contact_hand_world_ = hand_world(
                                            mapped_pose_at_frame(
                                                *database_,
                                                *candidate_,
                                                candidate_->contact_frame),
                                            affordance_->hand);
                                        has_previous_corrected_hand_ = false;
                                        corrected_path_clear_ = true;
                                        accumulate_runtime_clearance(
                                            corrected_path_clear_,
                                            has_previous_corrected_hand_,
                                            previous_corrected_hand_world_,
                                            hand_world(
                                                entry_blend_source_,
                                                affordance_->hand).position,
                                            *target_,
                                            *affordance_,
                                            false);
                                        entry_blend_elapsed_seconds_ = 0.0F;
                                        const float wall_time_to_contact =
                                            database_->time_to_contact.at(
                                                static_cast<size_t>(
                                                    candidate_->entry_frame)) /
                                            config_.playback.speed;
                                        commit_seconds_ = std::min({
                                            config_.playback.commit_horizon_seconds,
                                            wall_time_to_contact,
                                            config_.playback.maximum_alignment_seconds,
                                        });
                                        contact_evaluated_ = false;
                                        ever_attached_ = false;
                                        post_commit_failure_ = false;
                                        final_failure_frame_presented_ = false;
                                        carry_ready_ = false;
                                        carry_started_ = false;
                                        post_commit_reason_ = Reason::None;
                                        carry_.reset();
                                        state_ = RuntimeState::Align;
                                        diagnostics_.state = state_;
                                        diagnostics_.result =
                                            ResultCode::Accepted;
                                        diagnostics_.reason = Reason::None;
                                        diagnostics_.target = request.target;
                                        diagnostics_.object_state =
                                            ObjectState::Targeted;
                                        diagnostics_.affordance_id =
                                            request.affordance_id;
                                        diagnostics_.clip = candidate_->clip;
                                        diagnostics_.frame = player_->frame();
                                        diagnostics_.phase = player_->phase();
                                        diagnostics_.hand = affordance_->hand;
                                        diagnostics_.total_cost =
                                            candidate_->total_cost;
                                        diagnostics_.group_costs =
                                            candidate_->group_costs;
                                        diagnostics_.requested_root_correction_m =
                                            std::hypot(
                                                candidate_->entry_root_offset.x,
                                                candidate_->entry_root_offset.z);
                                        diagnostics_.applied_root_correction_m =
                                            diagnostics_.requested_root_correction_m;
                                        diagnostics_.requested_yaw_correction_radians =
                                            std::abs(candidate_->entry_yaw_offset);
                                        diagnostics_.applied_yaw_correction_radians =
                                            diagnostics_.requested_yaw_correction_radians;
                                        diagnostics_.playback_speed =
                                            config_.playback.speed;
                                    }
                                }
                            }
                        } catch (...) {
                            if (owns_reservation_) {
                                (void)registry_->release(
                                    request.target, request.request_id);
                            }
                            owns_reservation_ = false;
                            player_.reset();
                            event_player_.reset();
                            attachment_.reset();
                            carry_.reset();
                            candidate_.reset();
                            target_.reset();
                            affordance_.reset();
                            state_ = RuntimeState::Preflight;
                            diagnostics_.state = state_;
                            diagnostics_.result = ResultCode::None;
                            diagnostics_.reason = Reason::None;
                            diagnostics_.object_state = ObjectState::Free;
                            diagnostics_.attached = false;
                            throw;
                        }
                    }
                }
            }
        }
    } else if (state_ == RuntimeState::Align) {
        if (input.cancel_pressed) {
            if (owns_reservation_) {
                (void)registry_->release(
                    request_->target, request_->request_id);
            }
            owns_reservation_ = false;
            state_ = RuntimeState::Locomotion;
            diagnostics_.state = state_;
            diagnostics_.result = ResultCode::Cancelled;
            diagnostics_.reason = Reason::Cancelled;
            diagnostics_.object_state = ObjectState::Free;
            diagnostics_.attached = false;
            player_.reset();
            event_player_.reset();
            attachment_.reset();
            candidate_.reset();
            target_.reset();
            affordance_.reset();
        } else {
            const InteractionTarget* current = registry_->find(
                request_->target);
            if (current == nullptr || !target_.has_value() ||
                !exact(*current, *target_) ||
                !registry_->validate(
                    request_->target, request_->request_id)) {
                if (owns_reservation_) {
                    (void)registry_->release(
                        request_->target, request_->request_id);
                }
                owns_reservation_ = false;
                state_ = RuntimeState::Locomotion;
                diagnostics_.state = state_;
                diagnostics_.result = ResultCode::Rejected;
                diagnostics_.reason = Reason::TargetChanged;
                diagnostics_.object_state = ObjectState::Free;
                diagnostics_.attached = false;
            } else {
                player_->advance(input.dt);
                entry_blend_elapsed_seconds_ += input.dt;
                Pose sampled = player_->sample();
                if (player_->frame() <= candidate_->contact_frame) {
                    const IKResult ik = apply_reach_ik(
                        sampled,
                        player_->frame(),
                        *candidate_,
                        *affordance_,
                        target_hand_world_,
                        source_contact_hand_world_,
                        config_.ik);
                    diagnostics_.hand_position_error_m =
                        ik.position_error_m;
                    diagnostics_.hand_orientation_error_radians =
                        ik.orientation_error_radians;
                }
                pose_ = apply_entry_blend(
                    entry_blend_source_,
                    sampled,
                    entry_blend_elapsed_seconds_,
                    config_.playback.entry_blend_seconds);
                diagnostics_.frame = player_->frame();
                diagnostics_.phase = player_->phase();
                diagnostics_.applied_root_correction_m = length(
                    player_->entry_root_correction());
                diagnostics_.applied_yaw_correction_radians =
                    diagnostics_.requested_yaw_correction_radians *
                    correction_weight(
                        *candidate_, *player_, config_.playback.speed);
                if (player_->elapsed_seconds() >= commit_seconds_) {
                    state_ = RuntimeState::PickupReplay;
                    diagnostics_.state = state_;
                }
            }
        }
    } else if ((state_ == RuntimeState::PickupReplay ||
                state_ == RuntimeState::Hold) &&
               post_commit_failure_ && player_->finished() &&
               final_failure_frame_presented_) {
        const InteractionTarget* stored = registry_->find(request_->target);
        if (owns_reservation_ && stored != nullptr &&
            registry_->validate(
                request_->target, request_->request_id) &&
            (stored->state == ObjectState::Attached ||
             stored->state == ObjectState::Held)) {
            (void)registry_->replace_pose(
                request_->target.id, object_world_);
        } else if (owns_reservation_) {
            (void)registry_->release(
                request_->target, request_->request_id);
        }
        owns_reservation_ = false;
        state_ = RuntimeState::Locomotion;
        diagnostics_.state = state_;
        diagnostics_.result = ResultCode::Failed;
        diagnostics_.reason = post_commit_reason_;
        diagnostics_.object_state = ObjectState::Free;
        diagnostics_.attached = false;
        diagnostics_.recorded_carry = false;
        player_.reset();
        event_player_.reset();
        attachment_.reset();
        carry_.reset();
        candidate_.reset();
        target_.reset();
        affordance_.reset();
    } else if (state_ == RuntimeState::PickupReplay) {
        const float published_elapsed_seconds = player_->elapsed_seconds();
        if (!player_->finished()) player_->advance(input.dt);
        entry_blend_elapsed_seconds_ += input.dt;
        Pose sampled = player_->sample();
        if (!post_commit_failure_ &&
            player_->frame() <= candidate_->contact_frame) {
            const IKResult ik = apply_reach_ik(
                sampled,
                player_->frame(),
                *candidate_,
                *affordance_,
                target_hand_world_,
                source_contact_hand_world_,
                config_.ik);
            diagnostics_.hand_position_error_m = ik.position_error_m;
            diagnostics_.hand_orientation_error_radians =
                ik.orientation_error_radians;
        }
        pose_ = apply_entry_blend(
            entry_blend_source_,
            sampled,
            entry_blend_elapsed_seconds_,
            config_.playback.entry_blend_seconds);
        diagnostics_.frame = player_->frame();
        diagnostics_.phase = player_->phase();
        diagnostics_.applied_root_correction_m = length(
            player_->entry_root_correction());
        diagnostics_.applied_yaw_correction_radians =
            diagnostics_.requested_yaw_correction_radians *
            correction_weight(
                *candidate_, *player_, config_.playback.speed);
        drain_playback_events(published_elapsed_seconds);

        if (!post_commit_failure_ && ever_attached_ && player_->at_hold()) {
            state_ = RuntimeState::Hold;
            diagnostics_.state = state_;
        } else if (!post_commit_failure_ && player_->finished() &&
                   !contact_evaluated_) {
            post_commit_failure_ = true;
            post_commit_reason_ = Reason::ClipEnded;
            diagnostics_.result = ResultCode::Failed;
            diagnostics_.reason = post_commit_reason_;
            diagnostics_.attached = false;
        }
        if (post_commit_failure_ && player_->finished()) {
            final_failure_frame_presented_ = true;
        }
    } else if (state_ == RuntimeState::Hold) {
        if (carry_ready_) {
            begin_carry();
        } else {
            entry_blend_elapsed_seconds_ += input.dt;
            const float published_elapsed_seconds =
                player_->elapsed_seconds();
            const bool extending_final_pose = player_->finished();
            if (!extending_final_pose) player_->advance(input.dt);
            pose_ = apply_entry_blend(
                entry_blend_source_,
                player_->sample(),
                entry_blend_elapsed_seconds_,
                config_.playback.entry_blend_seconds);
            diagnostics_.frame = player_->frame();
            diagnostics_.phase = player_->phase();
            diagnostics_.applied_root_correction_m = length(
                player_->entry_root_correction());
            diagnostics_.applied_yaw_correction_radians = 0.0F;

            if (!post_commit_failure_) {
                drain_playback_events(published_elapsed_seconds);
            }

            if (post_commit_failure_) {
                if (player_->finished()) {
                    final_failure_frame_presented_ = true;
                }
            } else if (extending_final_pose) {
                const ContactMeasurement measurement = contact_measurement(
                    *database_,
                    pose_,
                    player_->frame(),
                    request_->target,
                    affordance_->hand,
                    target_hand_world_,
                    false,
                    true,
                    true);
                diagnostics_.hand_position_error_m =
                    measurement.position_error_m;
                diagnostics_.hand_orientation_error_radians =
                    measurement.orientation_error_radians;
                attachment_->update(measurement, input.dt);
                object_world_ = attachment_->object_world();
                if (attachment_->result() == ResultCode::Failed) {
                    post_commit_failure_ = true;
                    post_commit_reason_ = attachment_->reason();
                    diagnostics_.result = ResultCode::Failed;
                    diagnostics_.reason = post_commit_reason_;
                    diagnostics_.attached = false;
                    if (player_->finished()) {
                        final_failure_frame_presented_ = true;
                    }
                }
            }

            if (!post_commit_failure_) {
                if (attachment_->state() == ObjectState::Held &&
                    event_player_->at_hold()) {
                    carry_ready_ = true;
                    begin_carry();
                } else {
                    diagnostics_.object_state = ObjectState::Attached;
                    diagnostics_.attached = true;
                }
            }
        }
    } else if (state_ == RuntimeState::Carry) {
        if (input.reset_pressed) {
            const InteractionTarget* authoritative =
                registry_->find_by_id(request_->target.id);
            const bool may_restore = owns_reservation_ &&
                authoritative != nullptr &&
                authoritative->handle == request_->target &&
                registry_->validate(
                    request_->target, request_->request_id) &&
                authoritative->state == ObjectState::Held;
            TargetHandle reset{};
            if (may_restore) {
                reset = attachment_->reset(original_object_world_);
                object_world_ = original_object_world_;
            } else if (authoritative != nullptr) {
                object_world_ = authoritative->object_world;
            }
            owns_reservation_ = false;
            state_ = RuntimeState::Locomotion;
            diagnostics_.state = state_;
            diagnostics_.result = may_restore
                ? ResultCode::Reset
                : ResultCode::Failed;
            diagnostics_.reason = may_restore
                ? Reason::Reset
                : Reason::TargetChanged;
            if (may_restore) diagnostics_.target = reset;
            diagnostics_.object_state = may_restore
                ? ObjectState::Free
                : (authoritative == nullptr
                       ? ObjectState::Free
                       : authoritative->state);
            diagnostics_.attached = false;
            diagnostics_.recorded_carry = false;
            carry_.reset();
            player_.reset();
            event_player_.reset();
            attachment_.reset();
            candidate_.reset();
            target_.reset();
            affordance_.reset();
            request_.reset();
        } else {
            const InteractionTarget* authoritative = registry_->find_by_id(
                request_->target.id);
            const bool still_owns_held_target = owns_reservation_ &&
                authoritative != nullptr &&
                authoritative->handle == request_->target &&
                registry_->validate(
                    request_->target, request_->request_id) &&
                authoritative->state == ObjectState::Held;
            if (!still_owns_held_target) {
                if (authoritative != nullptr) {
                    object_world_ = authoritative->object_world;
                    diagnostics_.target = authoritative->handle;
                    diagnostics_.object_state = authoritative->state;
                } else {
                    diagnostics_.object_state = ObjectState::Free;
                }
                owns_reservation_ = false;
                state_ = RuntimeState::Locomotion;
                diagnostics_.state = state_;
                diagnostics_.result = ResultCode::Failed;
                diagnostics_.reason = Reason::TargetChanged;
                diagnostics_.attached = false;
                diagnostics_.recorded_carry = false;
                carry_.reset();
                player_.reset();
                event_player_.reset();
                attachment_.reset();
                candidate_.reset();
                target_.reset();
                affordance_.reset();
                request_.reset();
            } else {
                pose_ = carry_->update(input.locomotion, input.dt);
                object_world_ = carry_->object_world();
                diagnostics_.state = RuntimeState::Carry;
                diagnostics_.result = ResultCode::Succeeded;
                diagnostics_.reason = Reason::None;
                diagnostics_.object_state = ObjectState::Held;
                diagnostics_.attached = true;
                diagnostics_.recorded_carry = carry_->recorded();
            }
        }
    }

    RuntimeOutput output = passthrough(input, diagnostics_);
    output.object_world = object_world_;
    if (state_ == RuntimeState::Align ||
        state_ == RuntimeState::PickupReplay ||
        state_ == RuntimeState::Hold ||
        state_ == RuntimeState::Carry) {
        output.owns_pose = true;
        output.pose = pose_;
    }
    if (state_ == RuntimeState::PickupReplay ||
        state_ == RuntimeState::Hold) {
        output.suppress_steering = true;
    }
    return output;
}

}  // namespace interaction
