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
        left.object_profile_id != right.object_profile_id ||
        !exact(
            left.object_bounds.center_object,
            right.object_bounds.center_object) ||
        !exact(
            left.object_bounds.half_extents_object,
            right.object_bounds.half_extents_object) ||
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

bool exact(const ObjectLocalBounds& left, const ObjectLocalBounds& right) {
    return exact(left.center_object, right.center_object) &&
           exact(left.half_extents_object, right.half_extents_object);
}

bool exact(const Pose& left, const Pose& right) {
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

bool exact(const MatchCandidate& left, const MatchCandidate& right) {
    return left.clip == right.clip &&
           left.entry_frame == right.entry_frame &&
           left.contact_frame == right.contact_frame &&
           left.lift_frame == right.lift_frame &&
           left.hold_frame == right.hold_frame &&
           exact(left.scene_from_source, right.scene_from_source) &&
           exact(left.entry_root_offset, right.entry_root_offset) &&
           left.entry_yaw_offset == right.entry_yaw_offset &&
           left.total_cost == right.total_cost &&
           left.group_costs == right.group_costs;
}

bool exact(const PlaceTimingConfig& left, const PlaceTimingConfig& right) {
    return left.canonical_fps == right.canonical_fps &&
           left.playback_speed == right.playback_speed &&
           left.entry_blend_seconds == right.entry_blend_seconds &&
           left.reversed_commit_seconds == right.reversed_commit_seconds &&
           left.maximum_alignment_seconds ==
               right.maximum_alignment_seconds;
}

bool exact(const PlaceMatchConfig& left, const PlaceMatchConfig& right) {
    return left.maximum_entry_root_error_m ==
               right.maximum_entry_root_error_m &&
           left.maximum_entry_yaw_error_radians ==
               right.maximum_entry_yaw_error_radians;
}

bool exact(const IKConfig& left, const IKConfig& right) {
    return left.maximum_request_position_m ==
               right.maximum_request_position_m &&
           left.maximum_request_orientation_radians ==
               right.maximum_request_orientation_radians &&
           left.accepted_position_m == right.accepted_position_m &&
           left.accepted_orientation_radians ==
               right.accepted_orientation_radians &&
           left.damping == right.damping &&
           left.finite_difference_radians ==
               right.finite_difference_radians &&
           left.orientation_scale_m_per_radian ==
               right.orientation_scale_m_per_radian &&
           left.maximum_step_radians == right.maximum_step_radians &&
           left.maximum_iterations == right.maximum_iterations;
}

bool exact(const PlaceAffordance& left, const PlaceAffordance& right) {
    return left.id == right.id &&
           exact(left.object_in_surface, right.object_in_surface) &&
           exact(left.support_point_object, right.support_point_object) &&
           exact(
               left.approach_direction_surface,
               right.approach_direction_surface) &&
           left.clearance_radius == right.clearance_radius;
}

bool exact(const PlacementSurface& left, const PlacementSurface& right) {
    if (!(left.handle == right.handle) ||
        !exact(left.surface_world, right.surface_world) ||
        !exact(left.support_volume_world, right.support_volume_world) ||
        !exact(left.support_volume_size, right.support_volume_size) ||
        left.half_extent_x_m != right.half_extent_x_m ||
        left.half_extent_z_m != right.half_extent_z_m ||
        left.overhead_clearance_m != right.overhead_clearance_m ||
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

bool exact(const PlaceCandidate& left, const PlaceCandidate& right) {
    return left.mode == right.mode && left.source_id == right.source_id &&
           left.selection_id == right.selection_id &&
           exact(left.timing, right.timing) &&
           exact(left.match, right.match) && exact(left.ik, right.ik) &&
           left.clip == right.clip &&
           left.entry_frame == right.entry_frame &&
           left.commit_frame == right.commit_frame &&
           left.release_frame == right.release_frame &&
           left.stop_frame == right.stop_frame &&
           left.direction == right.direction &&
           exact(left.scene_from_source, right.scene_from_source) &&
           exact(left.staging_root_world, right.staging_root_world) &&
           exact(left.entry_root_offset, right.entry_root_offset) &&
           left.entry_yaw_offset == right.entry_yaw_offset &&
           left.total_cost == right.total_cost;
}

bool exact(const PlaceMatchInput& left, const PlaceMatchInput& right) {
    return left.pickup_database == right.pickup_database &&
           left.library == right.library &&
           left.held_target == right.held_target &&
           exact(left.pickup_candidate, right.pickup_candidate) &&
           exact(left.current_pose, right.current_pose) &&
           exact(left.current_object_world, right.current_object_world) &&
           left.held_object_profile_id == right.held_object_profile_id &&
           exact(left.held_object_bounds, right.held_object_bounds) &&
           exact(left.held_affordance, right.held_affordance) &&
           exact(left.surface, right.surface) &&
           exact(left.place_affordance, right.place_affordance) &&
           exact(left.object_dimensions, right.object_dimensions) &&
           exact(left.timing, right.timing) &&
           exact(left.match, right.match) && exact(left.ik, right.ik);
}

bool exact_held_metadata(
    const InteractionTarget& retained,
    const InteractionTarget& current) {
    if (retained.handle != current.handle ||
        !exact(retained.object_world, current.object_world) ||
        retained.object_profile_id != current.object_profile_id ||
        !exact(retained.object_bounds, current.object_bounds) ||
        !exact(retained.object_dimensions, current.object_dimensions) ||
        !exact(retained.table_world, current.table_world) ||
        !exact(retained.table_size, current.table_size) ||
        retained.owner_request != current.owner_request ||
        retained.affordances.size() != current.affordances.size()) {
        return false;
    }
    for (size_t index = 0; index < retained.affordances.size(); ++index) {
        if (!exact(retained.affordances[index], current.affordances[index])) {
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

float authored_hand_constraint_weight(
    int32_t frame,
    const MatchCandidate& candidate) {
    return std::clamp(
        static_cast<float>(frame - candidate.entry_frame) /
            static_cast<float>(
                candidate.contact_frame - candidate.entry_frame),
        0.0F,
        1.0F);
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
    const float reach_alpha = authored_hand_constraint_weight(
        frame, candidate);
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
    const PlaceController place_config_validator(config_.place, config_.ik);
    (void)place_config_validator;
    diagnostics_.state = state_;
    diagnostics_.playback_speed = config_.playback.speed;
    diagnostics_.pack_available = true;
}

InteractionRuntime::InteractionRuntime(
    const Database& database,
    const Features& features,
    TargetRegistry& registry,
    PlacementSurfaceRegistry& surface_registry,
    const PlaceMotionLibrary& place_library,
    RuntimeConfig config)
    : database_(&database),
      features_(&features),
      registry_(&registry),
      surface_registry_(&surface_registry),
      place_library_(&place_library),
      config_(config),
      state_(RuntimeState::Locomotion) {
    validate_runtime_config(config_);
    place_controller_.emplace(config_.place, config_.ik);
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

InteractionRuntime::PlaceMatchBuildResult
InteractionRuntime::make_place_match_input(
    SurfaceHandle surface,
    uint32_t affordance_id) const {
    PlaceMatchBuildResult result{};
    if (database_ == nullptr || features_ == nullptr || registry_ == nullptr ||
        surface_registry_ == nullptr || place_library_ == nullptr ||
        !place_controller_.has_value()) {
        result.reason = Reason::PackUnavailable;
        return result;
    }
    if (!request_.has_value() || !target_.has_value() ||
        !affordance_.has_value() || !candidate_.has_value() ||
        !attachment_.has_value() || !owns_reservation_) {
        result.reason = Reason::TargetChanged;
        return result;
    }

    const InteractionTarget* held = registry_->find(request_->target);
    if (held == nullptr || held->state != ObjectState::Held ||
        held->owner_request != request_->request_id ||
        !registry_->validate(request_->target, request_->request_id) ||
        attachment_->state() != ObjectState::Held ||
        !exact_held_metadata(*target_, *held)) {
        result.reason = Reason::TargetChanged;
        return result;
    }
    const GraspAffordance* grasp = registry_->find_affordance(
        held->handle, affordance_->id);
    if (grasp == nullptr || !exact(*grasp, *affordance_)) {
        result.reason = Reason::TargetChanged;
        return result;
    }

    const PlacementSurface* destination = surface_registry_->find(surface);
    if (destination == nullptr) {
        result.reason = surface.id != 0U &&
                surface_registry_->find_by_id(surface.id) != nullptr
            ? Reason::SurfaceChanged
            : Reason::SurfaceUnavailable;
        return result;
    }
    const PlaceAffordance* place_affordance =
        surface_registry_->find_affordance(surface, affordance_id);
    if (place_affordance == nullptr) {
        result.reason = Reason::SurfaceUnavailable;
        return result;
    }

    result.input.pickup_database = database_;
    result.input.library = place_library_;
    result.input.held_target = held->handle;
    result.input.pickup_candidate = *candidate_;
    result.input.current_pose = pose_;
    result.input.current_object_world = object_world_;
    result.input.held_object_profile_id = held->object_profile_id;
    result.input.held_object_bounds = held->object_bounds;
    result.input.held_affordance = *grasp;
    result.input.surface = *destination;
    result.input.place_affordance = *place_affordance;
    result.input.object_dimensions = held->object_dimensions;
    result.input.timing = config_.place.timing;
    result.input.match = config_.place.match;
    result.input.ik = config_.ik;
    result.accepted = true;
    result.reason = Reason::None;
    return result;
}

PlaceStagingPreview InteractionRuntime::preview_place(
    SurfaceHandle surface,
    uint32_t affordance_id) const {
    if (state_ != RuntimeState::Carry ||
        diagnostics_.object_state != ObjectState::Held ||
        !diagnostics_.attached) {
        PlaceStagingPreview rejected{};
        rejected.reason = Reason::OutOfRange;
        return rejected;
    }
    const PlaceMatchBuildResult built = make_place_match_input(
        surface, affordance_id);
    if (!built.accepted) {
        PlaceStagingPreview rejected{};
        rejected.reason = built.reason;
        return rejected;
    }
    return preview_place_motion(built.input);
}

void InteractionRuntime::reset_place_attempt() {
    place_request_.reset();
    frozen_place_input_.reset();
    active_place_input_.reset();
    frozen_place_preview_ = PlaceStagingPreview{};
    place_step_ = PlaceStep{};
    place_final_frame_presented_ = false;
    place_controller_.reset();
    if (surface_registry_ != nullptr && place_library_ != nullptr) {
        place_controller_.emplace(config_.place, config_.ik);
    }
}

void InteractionRuntime::update_place_diagnostics(const PlaceStep& step) {
    RuntimePlaceDiagnostics& place = diagnostics_.place;
    place.source_frame = step.source_frame;
    place.source_frame_exact = step.source_frame_exact;
    place.phase = step.phase;
    place.committed = step.committed;
    place.release_due = step.release_due;
    place.support_sweep_clear = step.support_sweep_clear;
    place.actual_fit = step.actual_fit;
    place.requested_root_correction_m =
        step.requested_root_correction_m;
    place.applied_root_correction_m = step.applied_root_correction_m;
    place.requested_yaw_correction_radians =
        step.requested_yaw_correction_radians;
    place.applied_yaw_correction_radians =
        step.applied_yaw_correction_radians;
    place.requested_hand_correction_m =
        step.requested_hand_correction_m;
    place.applied_hand_correction_m = step.applied_hand_correction_m;
    place.requested_hand_orientation_radians =
        step.requested_hand_orientation_radians;
    place.applied_hand_orientation_radians =
        step.applied_hand_orientation_radians;
    place.reason = step.reason;
    diagnostics_.hand_position_error_m = step.hand_position_error_m;
    diagnostics_.hand_orientation_error_radians =
        step.hand_orientation_error_radians;
    diagnostics_.frame = step.source_frame;

    if (place.preview_available) {
        const PlaceCandidate& candidate = place.preview.candidate;
        place.mode = candidate.mode;
        place.selection_id = candidate.selection_id;
        place.clip = candidate.clip;
        place.commit_frame = candidate.commit_frame;
        place.time_to_release_seconds = place.released
            ? 0.0F
            : static_cast<float>(std::abs(
                  static_cast<double>(candidate.release_frame) -
                  step.source_frame_exact) /
                  (static_cast<double>(kCanonicalFps) *
                   candidate.timing.playback_speed));
    }
    place.support_position_error_m = length(
        step.object_world.position - place.requested_goal_world.position);
    place.support_orientation_error_radians = quat_angle_between(
        step.object_world.rotation,
        place.requested_goal_world.rotation);
}

void InteractionRuntime::reconstruct_carry(
    const Pose& pose,
    Transform object_world,
    Reason reason,
    ResultCode result) {
    if (database_ == nullptr || features_ == nullptr ||
        !affordance_.has_value()) {
        throw std::logic_error(
            "interaction runtime cannot reconstruct carry");
    }
    carry_.reset();
    carry_.emplace(
        *database_,
        *features_,
        classify_carry_ranges(*database_, config_.carry),
        config_.carry,
        config_.ik);
    carry_->start(pose, affordance_->hand, *affordance_, object_world);
    carry_started_ = true;
    pose_ = pose;
    object_world_ = object_world;
    state_ = RuntimeState::Carry;
    diagnostics_.state = state_;
    diagnostics_.result = result;
    diagnostics_.reason = reason;
    diagnostics_.object_state = ObjectState::Held;
    diagnostics_.attached = true;
    diagnostics_.recorded_carry = false;
    diagnostics_.inactive_arm_targets_locomotion =
        carry_->inactive_arm_targets_locomotion();
    diagnostics_.inactive_arm_tracks_locomotion = false;
    diagnostics_.place.reason = reason;
    reset_place_attempt();
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
    diagnostics_.inactive_arm_targets_locomotion =
        carry_->inactive_arm_targets_locomotion();
    diagnostics_.inactive_arm_tracks_locomotion = false;
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
        diagnostics_.inactive_arm_targets_locomotion = false;
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
    } else if (state_ == RuntimeState::PlacePreflight) {
        auto reject_preflight = [this](Reason reason) {
            state_ = RuntimeState::Carry;
            diagnostics_.state = state_;
            diagnostics_.result = ResultCode::Rejected;
            diagnostics_.reason = reason;
            diagnostics_.object_state = ObjectState::Held;
            diagnostics_.attached = true;
            diagnostics_.place.reason = reason;
            place_request_.reset();
            frozen_place_input_.reset();
            active_place_input_.reset();
            frozen_place_preview_ = PlaceStagingPreview{};
            place_step_ = PlaceStep{};
            place_final_frame_presented_ = false;
        };

        if (!place_request_.has_value()) {
            reject_preflight(Reason::TargetUnavailable);
        } else if (!request_.has_value() ||
                   place_request_->held_target != request_->target) {
            reject_preflight(Reason::TargetChanged);
        } else if (place_request_->request_id == 0U) {
            reject_preflight(Reason::TargetUnavailable);
        } else {
            const PlaceMatchBuildResult built = make_place_match_input(
                place_request_->surface,
                place_request_->affordance_id);
            if (!built.accepted) {
                reject_preflight(built.reason);
            } else if (!frozen_place_input_.has_value()) {
                reject_preflight(
                    frozen_place_preview_.reason == Reason::None
                        ? Reason::TargetChanged
                        : frozen_place_preview_.reason);
            } else if (!exact(built.input, *frozen_place_input_)) {
                reject_preflight(
                    exact(built.input.surface, frozen_place_input_->surface)
                        ? Reason::TargetChanged
                        : Reason::SurfaceChanged);
            } else {
                const PlaceStagingPreview preview = preview_place_motion(
                    built.input);
                diagnostics_.place.preview_available = preview.accepted;
                diagnostics_.place.preview = preview;
                diagnostics_.place.selection_id =
                    preview.candidate.selection_id;
                diagnostics_.place.ik_config_fingerprint =
                    preview.ik_config_fingerprint;
                diagnostics_.place.candidate_certified = preview.accepted;
                const bool config_identity = preview.accepted &&
                    exact(built.input.timing, config_.place.timing) &&
                    exact(built.input.match, config_.place.match) &&
                    exact(built.input.ik, config_.ik) &&
                    exact(preview.ik, built.input.ik) &&
                    exact(preview.candidate.timing, built.input.timing) &&
                    exact(preview.candidate.match, built.input.match) &&
                    exact(preview.candidate.ik, built.input.ik) &&
                    preview.ik_config_fingerprint != 0U;
                diagnostics_.place.preflight_config_identity =
                    config_identity;
                if (!preview.accepted) {
                    reject_preflight(preview.reason);
                } else if (!config_identity) {
                    reject_preflight(Reason::CorrectionLimit);
                } else if (!frozen_place_preview_.accepted ||
                           !exact(
                               preview.candidate,
                               frozen_place_preview_.candidate) ||
                           preview.ik_config_fingerprint !=
                               frozen_place_preview_.ik_config_fingerprint ||
                           place_request_->selection_id !=
                               preview.candidate.selection_id) {
                    reject_preflight(Reason::TargetChanged);
                } else if (!preview.ready) {
                    reject_preflight(Reason::CorrectionLimit);
                } else if (!place_controller_.has_value()) {
                    reject_preflight(Reason::PackUnavailable);
                } else {
                    const PlaceBeginResult begun = place_controller_->begin(
                        PlaceBeginInput{built.input, preview.candidate});
                    if (!begun.accepted) {
                        reject_preflight(begun.reason);
                    } else {
                        active_place_input_ = built.input;
                        frozen_place_preview_ = preview;
                        carry_.reset();
                        carry_started_ = false;
                        place_step_ = PlaceStep{};
                        place_step_.pose = pose_;
                        place_step_.object_world = object_world_;
                        place_step_.phase = PlacePhase::Align;
                        place_step_.source_frame =
                            preview.candidate.entry_frame;
                        place_step_.source_frame_exact =
                            preview.candidate.entry_frame;
                        place_step_.support_sweep_clear = true;
                        state_ = RuntimeState::PlaceAlign;
                        diagnostics_.state = state_;
                        diagnostics_.result = ResultCode::Accepted;
                        diagnostics_.reason = Reason::None;
                        diagnostics_.object_state = ObjectState::Held;
                        diagnostics_.attached = true;
                        diagnostics_.recorded_carry = false;
                        diagnostics_.inactive_arm_targets_locomotion = false;
                        diagnostics_.inactive_arm_tracks_locomotion = false;
                        diagnostics_.place.reason = Reason::None;
                        diagnostics_.place.surface =
                            place_request_->surface;
                        diagnostics_.place.affordance_id =
                            place_request_->affordance_id;
                        diagnostics_.place.mode = preview.candidate.mode;
                        diagnostics_.place.selection_id =
                            preview.candidate.selection_id;
                        diagnostics_.place.clip = preview.candidate.clip;
                        diagnostics_.place.commit_frame =
                            preview.candidate.commit_frame;
                        diagnostics_.place.effective_ik = config_.ik;
                        diagnostics_.place.requested_goal_world =
                            placement_goal_world(
                                built.input.surface,
                                built.input.place_affordance
                                    .object_in_surface);
                        diagnostics_.place.requested_fit =
                            evaluate_placement_fit(
                                built.input.surface,
                                built.input.place_affordance,
                                built.input.held_object_bounds);
                        diagnostics_.clip = preview.candidate.clip;
                        update_place_diagnostics(place_step_);
                    }
                }
            }
        }
    } else if (state_ == RuntimeState::PlaceAlign ||
               state_ == RuntimeState::PlaceReplay) {
        auto recover_or_publish_external_failure =
            [this](const PlaceStep& step, Reason reason, ResultCode result) {
                const InteractionTarget* held =
                    request_.has_value() && registry_ != nullptr
                    ? registry_->find(request_->target)
                    : nullptr;
                const bool still_held = request_.has_value() &&
                    held != nullptr && held->state == ObjectState::Held &&
                    held->owner_request == request_->request_id &&
                    registry_->validate(
                        request_->target, request_->request_id);
                if (still_held) {
                    reconstruct_carry(
                        step.pose, step.object_world, reason, result);
                    return;
                }

                const InteractionTarget* authoritative =
                    request_.has_value() && registry_ != nullptr
                    ? registry_->find_by_id(request_->target.id)
                    : nullptr;
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
                diagnostics_.inactive_arm_targets_locomotion = false;
                diagnostics_.inactive_arm_tracks_locomotion = false;
                diagnostics_.place.reason = Reason::TargetChanged;
                carry_.reset();
                player_.reset();
                event_player_.reset();
                attachment_.reset();
                candidate_.reset();
                target_.reset();
                affordance_.reset();
                request_.reset();
                reset_place_attempt();
            };

        if (!place_controller_.has_value() ||
            !active_place_input_.has_value() ||
            !place_request_.has_value() || !attachment_.has_value()) {
            PlaceStep failed = place_step_;
            failed.recover_to_carry = true;
            failed.reason = Reason::PackUnavailable;
            recover_or_publish_external_failure(
                failed, failed.reason, ResultCode::Failed);
        } else {
            PlaceStep step = place_controller_->update(input.dt);
            if (input.cancel_pressed) {
                step = place_controller_->cancel();
            }
            place_step_ = step;
            pose_ = step.pose;
            object_world_ = step.object_world;
            update_place_diagnostics(step);

            if (step.recover_to_carry) {
                recover_or_publish_external_failure(
                    step,
                    step.reason,
                    step.reason == Reason::Cancelled
                        ? ResultCode::Cancelled
                        : ResultCode::Failed);
            } else if (step.release_due) {
                Reason release_failure = Reason::None;
                const PlacementSurface* surface =
                    surface_registry_ == nullptr
                    ? nullptr
                    : surface_registry_->find(place_request_->surface);
                const PlaceAffordance* place_affordance =
                    surface_registry_ == nullptr
                    ? nullptr
                    : surface_registry_->find_affordance(
                          place_request_->surface,
                          place_request_->affordance_id);
                if (surface == nullptr || place_affordance == nullptr ||
                    !exact(*surface, active_place_input_->surface) ||
                    !exact(
                        *place_affordance,
                        active_place_input_->place_affordance)) {
                    release_failure = Reason::SurfaceChanged;
                }

                PlacementFit actual{};
                if (release_failure == Reason::None) {
                    try {
                        actual = evaluate_actual_placement_fit(
                            *surface,
                            *place_affordance,
                            step.object_world,
                            active_place_input_->held_object_bounds);
                    } catch (const std::exception&) {
                        actual.reason = Reason::PlacementOutOfBounds;
                    }
                    if (!actual.accepted) {
                        release_failure = actual.reason == Reason::None
                            ? Reason::PlacementOutOfBounds
                            : actual.reason;
                    }
                }
                if (release_failure == Reason::None &&
                    !step.support_sweep_clear) {
                    release_failure = Reason::BlockedPath;
                }
                diagnostics_.place.actual_fit = actual;

                if (release_failure != Reason::None) {
                    step.reason = release_failure;
                    step.actual_fit = actual;
                    update_place_diagnostics(step);
                    recover_or_publish_external_failure(
                        step, release_failure, ResultCode::Failed);
                } else {
                    const std::optional<TargetHandle> placed =
                        attachment_->commit_place(
                            step.object_world,
                            PlacedSupportContext{
                                surface->support_volume_world,
                                surface->support_volume_size});
                    if (!placed.has_value()) {
                        recover_or_publish_external_failure(
                            step, Reason::TargetChanged, ResultCode::Failed);
                    } else {
                        place_controller_->acknowledge_release(
                            step.object_world);
                        owns_reservation_ = false;
                        diagnostics_.target = *placed;
                        diagnostics_.object_state = ObjectState::Free;
                        diagnostics_.attached = false;
                        diagnostics_.result = ResultCode::Succeeded;
                        diagnostics_.reason = Reason::None;
                        diagnostics_.place.released = true;
                        diagnostics_.place.release_due = false;
                        diagnostics_.place.reason = Reason::None;
                        diagnostics_.place.actual_fit = actual;
                        diagnostics_.place.requested_root_correction_m = 0.0F;
                        diagnostics_.place.applied_root_correction_m = 0.0F;
                        diagnostics_.place.requested_yaw_correction_radians =
                            0.0F;
                        diagnostics_.place.applied_yaw_correction_radians =
                            0.0F;
                        diagnostics_.place.requested_hand_correction_m = 0.0F;
                        diagnostics_.place.applied_hand_correction_m = 0.0F;
                        diagnostics_.place
                            .requested_hand_orientation_radians = 0.0F;
                        diagnostics_.place
                            .applied_hand_orientation_radians = 0.0F;
                        state_ = RuntimeState::PlaceRelease;
                        diagnostics_.state = state_;
                        place_final_frame_presented_ = false;
                    }
                }
            } else {
                state_ = step.committed
                    ? RuntimeState::PlaceReplay
                    : RuntimeState::PlaceAlign;
                diagnostics_.state = state_;
                diagnostics_.result = ResultCode::Accepted;
                diagnostics_.reason = Reason::None;
                diagnostics_.object_state = ObjectState::Held;
                diagnostics_.attached = true;
                diagnostics_.place.reason = Reason::None;
            }
        }
    } else if (state_ == RuntimeState::PlaceRelease) {
        if (place_final_frame_presented_) {
            state_ = RuntimeState::Locomotion;
            diagnostics_.state = state_;
            diagnostics_.result = ResultCode::Succeeded;
            diagnostics_.reason = Reason::None;
            diagnostics_.object_state = ObjectState::Free;
            diagnostics_.attached = false;
            diagnostics_.recorded_carry = false;
            diagnostics_.inactive_arm_targets_locomotion = false;
            diagnostics_.inactive_arm_tracks_locomotion = false;
            carry_.reset();
            player_.reset();
            event_player_.reset();
            attachment_.reset();
            candidate_.reset();
            target_.reset();
            affordance_.reset();
            request_.reset();
            reset_place_attempt();
        } else if (!place_controller_.has_value()) {
            throw std::logic_error(
                "interaction runtime place release controller is missing");
        } else {
            place_step_ = place_controller_->update(input.dt);
            pose_ = place_step_.pose;
            object_world_ = place_step_.object_world;
            update_place_diagnostics(place_step_);
            diagnostics_.state = RuntimeState::PlaceRelease;
            diagnostics_.result = ResultCode::Succeeded;
            diagnostics_.reason = Reason::None;
            diagnostics_.object_state = ObjectState::Free;
            diagnostics_.attached = false;
            diagnostics_.place.released = true;
            diagnostics_.place.reason = Reason::None;
            if (place_step_.retract_finished) {
                place_final_frame_presented_ = true;
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
            diagnostics_.inactive_arm_targets_locomotion = false;
            diagnostics_.inactive_arm_tracks_locomotion = false;
            carry_.reset();
            player_.reset();
            event_player_.reset();
            attachment_.reset();
            candidate_.reset();
            target_.reset();
            affordance_.reset();
            request_.reset();
            reset_place_attempt();
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
                diagnostics_.inactive_arm_targets_locomotion = false;
                diagnostics_.inactive_arm_tracks_locomotion = false;
                carry_.reset();
                player_.reset();
                event_player_.reset();
                attachment_.reset();
                candidate_.reset();
                target_.reset();
                affordance_.reset();
                request_.reset();
                reset_place_attempt();
            } else if (input.interact_pressed) {
                place_request_ = input.place_request;
                frozen_place_input_.reset();
                active_place_input_.reset();
                frozen_place_preview_ = PlaceStagingPreview{};
                diagnostics_.place = RuntimePlaceDiagnostics{};
                diagnostics_.place.effective_ik = config_.ik;
                if (place_request_.has_value()) {
                    diagnostics_.place.surface = place_request_->surface;
                    diagnostics_.place.affordance_id =
                        place_request_->affordance_id;
                    const PlaceMatchBuildResult built = make_place_match_input(
                        place_request_->surface,
                        place_request_->affordance_id);
                    if (built.accepted) {
                        frozen_place_input_ = built.input;
                        frozen_place_preview_ = preview_place_motion(
                            built.input);
                        diagnostics_.place.preview_available =
                            frozen_place_preview_.accepted;
                        diagnostics_.place.preview = frozen_place_preview_;
                        diagnostics_.place.candidate_certified =
                            frozen_place_preview_.accepted;
                        diagnostics_.place.mode =
                            frozen_place_preview_.candidate.mode;
                        diagnostics_.place.selection_id =
                            frozen_place_preview_.candidate.selection_id;
                        diagnostics_.place.ik_config_fingerprint =
                            frozen_place_preview_.ik_config_fingerprint;
                        diagnostics_.place.requested_goal_world =
                            placement_goal_world(
                                built.input.surface,
                                built.input.place_affordance
                                    .object_in_surface);
                        diagnostics_.place.requested_fit =
                            evaluate_placement_fit(
                                built.input.surface,
                                built.input.place_affordance,
                                built.input.held_object_bounds);
                    } else {
                        frozen_place_preview_.reason = built.reason;
                        diagnostics_.place.preview = frozen_place_preview_;
                        diagnostics_.place.reason = built.reason;
                    }
                }
                state_ = RuntimeState::PlacePreflight;
                diagnostics_.state = state_;
                diagnostics_.result = ResultCode::Accepted;
                diagnostics_.reason = Reason::None;
                diagnostics_.object_state = ObjectState::Held;
                diagnostics_.attached = true;
            } else {
                const Pose next_pose = carry_->update(
                    input.locomotion, input.dt);
                const Transform next_object_world = carry_->object_world();
                const bool next_recorded_carry = carry_->recorded();
                const bool next_inactive_arm_tracks_locomotion =
                    carry_->inactive_arm_tracks_locomotion();
                pose_ = next_pose;
                object_world_ = next_object_world;
                diagnostics_.state = RuntimeState::Carry;
                diagnostics_.result = ResultCode::Succeeded;
                diagnostics_.reason = Reason::None;
                diagnostics_.object_state = ObjectState::Held;
                diagnostics_.attached = true;
                diagnostics_.recorded_carry = next_recorded_carry;
                diagnostics_.inactive_arm_targets_locomotion =
                    carry_->inactive_arm_targets_locomotion();
                diagnostics_.inactive_arm_tracks_locomotion =
                    next_inactive_arm_tracks_locomotion;
            }
        }
    }

    diagnostics_.hand_constraint_weight = 0.0F;
    if (diagnostics_.result != ResultCode::Failed &&
        candidate_.has_value() && player_.has_value() &&
        (state_ == RuntimeState::Align ||
         state_ == RuntimeState::PickupReplay ||
         state_ == RuntimeState::Hold ||
         state_ == RuntimeState::Carry)) {
        diagnostics_.hand_constraint_weight =
            authored_hand_constraint_weight(
                player_->frame(), *candidate_);
    }

    RuntimeOutput output = passthrough(input, diagnostics_);
    output.object_world = object_world_;
    if (state_ == RuntimeState::Align ||
        state_ == RuntimeState::PickupReplay ||
        state_ == RuntimeState::Hold ||
        state_ == RuntimeState::Carry ||
        state_ == RuntimeState::PlacePreflight ||
        state_ == RuntimeState::PlaceAlign ||
        state_ == RuntimeState::PlaceReplay ||
        state_ == RuntimeState::PlaceRelease) {
        output.owns_pose = true;
        output.pose = pose_;
    }
    if (state_ == RuntimeState::PickupReplay ||
        state_ == RuntimeState::Hold ||
        state_ == RuntimeState::PlacePreflight ||
        state_ == RuntimeState::PlaceAlign ||
        state_ == RuntimeState::PlaceReplay ||
        state_ == RuntimeState::PlaceRelease) {
        output.suppress_steering = true;
    }
    return output;
}

}  // namespace interaction
