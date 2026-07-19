#include "interaction_place_controller.h"

#include "interaction_place_collision.h"
#include "interaction_rotation_gate.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace interaction {
namespace {

constexpr float kCanonicalDt = 0.04F;
constexpr float kCanonicalFps = 25.0F;
constexpr float kMinimumPlaybackSpeed = 0.85F;
constexpr float kMaximumPlaybackSpeed = 1.15F;
constexpr float kMaximumEntryRootM = 0.25F;
constexpr float kMaximumEntryYawRadians = 0.436332313F;
constexpr float kMaximumRequestPositionM = 0.12F;
constexpr float kMaximumRequestOrientationRadians = 0.436332313F;
constexpr float kMaximumReleasePositionM = 0.02F;
constexpr float kMaximumReleaseOrientationRadians = 0.174532925F;
constexpr float kMinimumRotationNorm = 1.0e-12F;
constexpr float kRotationUnitTolerance = 1.0e-3F;
constexpr size_t kRoot = static_cast<size_t>(g1_skeleton::Simulation);

bool finite(float value) {
    uint32_t bits = 0U;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    return (bits & 0x7f800000U) != 0x7f800000U;
}

bool finite(vec3 value) {
    return finite(value.x) && finite(value.y) && finite(value.z);
}

bool finite(quat value) {
    return finite(value.w) && finite(value.x) && finite(value.y) &&
           finite(value.z);
}

quat normalized(quat value) {
    const double norm = std::sqrt(
        static_cast<double>(value.w) * value.w +
        static_cast<double>(value.x) * value.x +
        static_cast<double>(value.y) * value.y +
        static_cast<double>(value.z) * value.z);
    if (!(norm > kMinimumRotationNorm) ||
        !rotation_gate::finite_bits(norm)) {
        return quat();
    }
    return value * static_cast<float>(1.0 / norm);
}

bool valid_rotation(quat value) {
    if (!finite(value)) return false;
    const double squared_norm =
        static_cast<double>(value.w) * value.w +
        static_cast<double>(value.x) * value.x +
        static_cast<double>(value.y) * value.y +
        static_cast<double>(value.z) * value.z;
    return squared_norm >
               static_cast<double>(kMinimumRotationNorm) *
                   kMinimumRotationNorm &&
           std::abs(std::sqrt(squared_norm) - 1.0) <=
               kRotationUnitTolerance;
}

bool valid_transform(Transform value) {
    return finite(value.position) && valid_rotation(value.rotation);
}

bool same_float(float left, float right) {
    if (left == 0.0F && right == 0.0F) return true;
    uint32_t left_bits = 0U;
    uint32_t right_bits = 0U;
    std::memcpy(&left_bits, &left, sizeof(left_bits));
    std::memcpy(&right_bits, &right, sizeof(right_bits));
    return left_bits == right_bits;
}

bool same_vec(vec3 left, vec3 right) {
    return same_float(left.x, right.x) && same_float(left.y, right.y) &&
           same_float(left.z, right.z);
}

std::array<float, 4> canonical_rotation(quat value) {
    value = normalized(value);
    std::array<float, 4> components = {
        value.w, value.x, value.y, value.z};
    bool negate = false;
    for (float component : components) {
        if (component == 0.0F) continue;
        negate = component < 0.0F;
        break;
    }
    if (negate) {
        for (float& component : components) component = -component;
    }
    for (float& component : components) {
        if (component == 0.0F) component = 0.0F;
    }
    return components;
}

bool same_rotation(quat left, quat right) {
    if (!valid_rotation(left) || !valid_rotation(right)) return false;
    const std::array<float, 4> canonical_left = canonical_rotation(left);
    const std::array<float, 4> canonical_right = canonical_rotation(right);
    for (size_t component = 0; component < canonical_left.size(); ++component) {
        if (!same_float(canonical_left[component], canonical_right[component])) {
            return false;
        }
    }
    return true;
}

bool same_transform(Transform left, Transform right) {
    return same_vec(left.position, right.position) &&
           same_rotation(left.rotation, right.rotation);
}

bool same_timing(
    const PlaceTimingConfig& left,
    const PlaceTimingConfig& right) {
    return same_float(left.canonical_fps, right.canonical_fps) &&
           same_float(left.playback_speed, right.playback_speed) &&
           same_float(left.entry_blend_seconds, right.entry_blend_seconds) &&
           same_float(
               left.reversed_commit_seconds,
               right.reversed_commit_seconds) &&
           same_float(
               left.maximum_alignment_seconds,
               right.maximum_alignment_seconds);
}

bool same_match(const PlaceMatchConfig& left, const PlaceMatchConfig& right) {
    return same_float(
               left.maximum_entry_root_error_m,
               right.maximum_entry_root_error_m) &&
           same_float(
               left.maximum_entry_yaw_error_radians,
               right.maximum_entry_yaw_error_radians);
}

bool same_ik(const IKConfig& left, const IKConfig& right) {
    return same_float(
               left.maximum_request_position_m,
               right.maximum_request_position_m) &&
           same_float(
               left.maximum_request_orientation_radians,
               right.maximum_request_orientation_radians) &&
           same_float(left.accepted_position_m, right.accepted_position_m) &&
           same_float(
               left.accepted_orientation_radians,
               right.accepted_orientation_radians) &&
           same_float(left.damping, right.damping) &&
           same_float(
               left.finite_difference_radians,
               right.finite_difference_radians) &&
           same_float(
               left.orientation_scale_m_per_radian,
               right.orientation_scale_m_per_radian) &&
           same_float(
               left.maximum_step_radians,
               right.maximum_step_radians) &&
           left.maximum_iterations == right.maximum_iterations;
}

bool same_candidate(
    const PlaceCandidate& left,
    const PlaceCandidate& right) {
    return left.mode == right.mode &&
           left.source_id == right.source_id &&
           left.selection_id == right.selection_id &&
           same_float(
               left.source_support_height_m,
               right.source_support_height_m) &&
           same_float(
               left.requested_support_height_m,
               right.requested_support_height_m) &&
           same_float(
               left.target_support_height_m,
               right.target_support_height_m) &&
           same_float(
               left.requested_vertical_correction_m,
               right.requested_vertical_correction_m) &&
           same_timing(left.timing, right.timing) &&
           same_match(left.match, right.match) &&
           same_ik(left.ik, right.ik) &&
           left.clip == right.clip &&
           left.entry_frame == right.entry_frame &&
           left.commit_frame == right.commit_frame &&
           left.release_frame == right.release_frame &&
           left.stop_frame == right.stop_frame &&
           left.direction == right.direction &&
           same_transform(left.scene_from_source, right.scene_from_source) &&
           same_transform(left.staging_root_world, right.staging_root_world) &&
           same_vec(left.entry_root_offset, right.entry_root_offset) &&
           same_float(left.entry_yaw_offset, right.entry_yaw_offset) &&
           same_float(left.total_cost, right.total_cost);
}

bool valid_timing(const PlaceTimingConfig& timing) {
    return finite(timing.canonical_fps) &&
           finite(timing.playback_speed) &&
           finite(timing.entry_blend_seconds) &&
           finite(timing.reversed_commit_seconds) &&
           finite(timing.maximum_alignment_seconds) &&
           timing.canonical_fps == kCanonicalFps &&
           timing.playback_speed >= kMinimumPlaybackSpeed &&
           timing.playback_speed <= kMaximumPlaybackSpeed &&
           timing.entry_blend_seconds > 0.0F &&
           timing.reversed_commit_seconds > 0.0F &&
           timing.maximum_alignment_seconds > 0.0F &&
           timing.maximum_alignment_seconds <= 1.0F &&
           timing.maximum_alignment_seconds >= timing.entry_blend_seconds &&
           timing.maximum_alignment_seconds >= timing.reversed_commit_seconds;
}

bool valid_match(const PlaceMatchConfig& match) {
    return finite(match.maximum_entry_root_error_m) &&
           finite(match.maximum_entry_yaw_error_radians) &&
           match.maximum_entry_root_error_m > 0.0F &&
           match.maximum_entry_yaw_error_radians > 0.0F &&
           match.maximum_entry_root_error_m <= kMaximumEntryRootM &&
           match.maximum_entry_yaw_error_radians <=
               kMaximumEntryYawRadians;
}

bool valid_ik(const IKConfig& config) {
    return finite(config.maximum_request_position_m) &&
           finite(config.maximum_request_orientation_radians) &&
           finite(config.accepted_position_m) &&
           finite(config.accepted_orientation_radians) &&
           finite(config.damping) &&
           finite(config.finite_difference_radians) &&
           finite(config.orientation_scale_m_per_radian) &&
           finite(config.maximum_step_radians) &&
           config.maximum_request_position_m >= 0.0F &&
           config.maximum_request_orientation_radians >= 0.0F &&
           config.accepted_position_m >= 0.0F &&
           config.accepted_orientation_radians >= 0.0F &&
           config.damping > 0.0F &&
           config.finite_difference_radians > 0.0F &&
           config.orientation_scale_m_per_radian > 0.0F &&
           config.maximum_step_radians > 0.0F &&
           config.maximum_iterations >= 0 &&
           config.maximum_request_position_m <= kMaximumRequestPositionM &&
           config.maximum_request_orientation_radians <=
               kMaximumRequestOrientationRadians;
}

bool valid_config(const PlaceControllerConfig& config, const IKConfig& ik) {
    return valid_timing(config.timing) && valid_match(config.match) &&
           valid_ik(ik) && finite(config.release_position_m) &&
           config.release_position_m > 0.0F &&
           config.release_position_m <= kMaximumReleasePositionM &&
           finite(config.release_orientation_radians) &&
           config.release_orientation_radians > 0.0F &&
           config.release_orientation_radians <=
               kMaximumReleaseOrientationRadians;
}

size_t hand_bone(Hand hand) {
    return hand == Hand::Left
        ? static_cast<size_t>(g1_skeleton::LeftWrist)
        : static_cast<size_t>(g1_skeleton::RightWrist);
}

Transform hand_world(const Pose& pose, Hand hand) {
    const WorldPose world = world_pose(pose);
    const size_t bone = hand_bone(hand);
    return {world.positions[bone], normalized(world.rotations[bone])};
}

rotation_gate::Rotation hand_world_rotation_evidence(
    const Pose& pose,
    Hand hand,
    const rotation_gate::Rotation& root_rotation_evidence) {
    return world_rotation_evidence(
        pose, hand_bone(hand), root_rotation_evidence);
}

Transform hand_derived_object(
    const Pose& pose,
    const GraspAffordance& affordance) {
    Transform hand = hand_world(pose, affordance.hand);
    hand.rotation = normalized(hand.rotation);
    return compose(hand, inverse(affordance.hand_in_object));
}

double distance(vec3 left, vec3 right) {
    const double x = static_cast<double>(left.x) - right.x;
    const double y = static_cast<double>(left.y) - right.y;
    const double z = static_cast<double>(left.z) - right.z;
    return std::sqrt(x * x + y * y + z * z);
}

float smoothstep(float value) {
    value = std::clamp(value, 0.0F, 1.0F);
    return value * value * (3.0F - 2.0F * value);
}

float source_progress(
    const PlaceCandidate& candidate,
    double source_frame) {
    const double numerator =
        static_cast<double>(candidate.direction) *
        (source_frame - candidate.entry_frame);
    const double denominator =
        static_cast<double>(candidate.direction) *
        (static_cast<double>(candidate.release_frame) -
         candidate.entry_frame);
    if (!(denominator > 0.0) ||
        !rotation_gate::finite_bits(source_frame)) {
        return 0.0F;
    }
    return std::clamp(
        static_cast<float>(numerator / denominator),
        0.0F,
        1.0F);
}

float replay_progress(
    const PlaceCandidate& candidate,
    double source_frame,
    bool committed) {
    if (!committed) return 0.0F;
    const double numerator =
        static_cast<double>(candidate.direction) *
        (source_frame - candidate.commit_frame);
    const double denominator =
        static_cast<double>(candidate.direction) *
        (static_cast<double>(candidate.release_frame) -
         candidate.commit_frame);
    if (!(denominator > 0.0) ||
        !rotation_gate::finite_bits(source_frame)) {
        return 0.0F;
    }
    return std::clamp(
        static_cast<float>(numerator / denominator),
        0.0F,
        1.0F);
}

Pose corrected_source_pose(
    Pose source,
    const PlaceBeginInput& begin,
    double source_frame,
    int32_t output_tick,
    int32_t entry_blend_ticks) {
    const float progress = source_progress(begin.candidate, source_frame);
    const float root_weight = 1.0F - smoothstep(progress);
    source.positions[kRoot] = source.positions[kRoot] +
        root_weight * begin.candidate.entry_root_offset;
    const quat yaw = quat_from_angle_axis(
        root_weight * begin.candidate.entry_yaw_offset,
        vec3(0.0F, 1.0F, 0.0F));
    source.rotations[kRoot] = normalized(quat_mul(yaw, source.rotations[kRoot]));

    const float blend_progress = std::clamp(
        static_cast<float>(output_tick) /
            static_cast<float>(entry_blend_ticks),
        0.0F,
        1.0F);
    if (blend_progress >= 1.0F) return source;
    return interpolate_pose(
        begin.match_input.current_pose,
        source,
        smoothstep(blend_progress));
}

rotation_gate::Rotation corrected_root_rotation_evidence(
    const Pose& corrected_pose,
    const rotation_gate::Rotation& mapped_source_root_evidence,
    const PlaceBeginInput& begin,
    double source_frame,
    int32_t output_tick,
    int32_t entry_blend_ticks) {
    const float blend_progress = std::clamp(
        static_cast<float>(output_tick) /
            static_cast<float>(entry_blend_ticks),
        0.0F,
        1.0F);
    if (blend_progress < 1.0F) {
        return rotation_gate::from_quat(
            corrected_pose.rotations[kRoot]);
    }
    const float progress = source_progress(begin.candidate, source_frame);
    const float root_weight = 1.0F - smoothstep(progress);
    const quat yaw = quat_from_angle_axis(
        root_weight * begin.candidate.entry_yaw_offset,
        vec3(0.0F, 1.0F, 0.0F));
    return rotation_gate::multiply(
        rotation_gate::from_quat(yaw),
        mapped_source_root_evidence);
}

struct WeightedHandTarget {
    Transform value{};
    quat correction{};
};

WeightedHandTarget weighted_hand_target(
    Transform current,
    vec3 release_translation,
    quat release_rotation,
    float progress) {
    progress = std::clamp(progress, 0.0F, 1.0F);
    const quat weighted_rotation = quat_nlerp_shortest(
        quat(), release_rotation, progress);
    return {
        Transform{
            current.position + progress * release_translation,
            normalized(quat_mul(weighted_rotation, current.rotation)),
        },
        weighted_rotation,
    };
}

PlaceStep recovery_step(
    const Pose& pose,
    Transform object_world,
    const PlaceStep& previous,
    Reason reason) {
    PlaceStep step = previous;
    step.pose = pose;
    step.object_world = object_world;
    step.release_due = false;
    step.retract_finished = false;
    step.recover_to_carry = true;
    step.reason = reason;
    step.support_sweep_clear = false;
    return step;
}

void copy_candidate_provenance(
    PlaceStep& step,
    const PlaceCandidate& candidate) {
    step.source_id = candidate.source_id;
    step.source_support_height_m = candidate.source_support_height_m;
    step.requested_support_height_m =
        candidate.requested_support_height_m;
    step.target_support_height_m = candidate.target_support_height_m;
    step.requested_vertical_correction_m =
        candidate.requested_vertical_correction_m;
}

}  // namespace

PlaceController::PlaceController(
    PlaceControllerConfig config,
    IKConfig ik_config)
    : config_(config), ik_config_(ik_config) {
    if (!valid_config(config_, ik_config_)) {
        throw std::invalid_argument("invalid place controller configuration");
    }
}

PlaceBeginResult PlaceController::begin(const PlaceBeginInput& input) {
    const bool reusable_terminal = started_ &&
        (recovering_ ||
         (release_acknowledged_ && player_.finished()));
    if (started_ && !reusable_terminal) {
        return {false, Reason::TargetChanged};
    }
    if (!same_timing(input.match_input.timing, config_.timing) ||
        !same_timing(input.candidate.timing, config_.timing) ||
        !same_match(input.match_input.match, config_.match) ||
        !same_match(input.candidate.match, config_.match) ||
        !same_ik(input.match_input.ik, ik_config_) ||
        !same_ik(input.candidate.ik, ik_config_)) {
        return {false, Reason::CorrectionLimit};
    }

    const PlaceResult selected = select_place_motion(input.match_input);
    if (!selected.accepted) return {false, selected.reason};
    if (!same_candidate(input.candidate, selected.candidate)) {
        return {false, Reason::TargetChanged};
    }

    try {
        PlacePlayer player;
        player.start_validated(input.candidate, input.match_input);
        PlacePlayer release_probe = player;
        int32_t guard = 0;
        while (!release_probe.release_due() && guard++ < 100000) {
            release_probe.advance(kCanonicalDt);
        }
        if (!release_probe.release_due()) return {false, Reason::ClipEnded};
        const PlaceSample release_sample = release_probe.sample();
        const Transform goal_object = placement_goal_world(
            input.match_input.surface,
            input.match_input.place_affordance.object_in_surface);
        const Transform goal_hand = compose(
            goal_object,
            input.match_input.held_affordance.hand_in_object);
        const rotation_gate::Rotation goal_object_rotation_evidence =
            rotation_gate::multiply(
                rotation_gate::from_quat(
                    input.match_input.surface.surface_world.rotation),
                rotation_gate::from_quat(
                    input.match_input.place_affordance
                        .object_in_surface.rotation));
        const rotation_gate::Rotation goal_hand_rotation_evidence =
            rotation_gate::multiply(
                goal_object_rotation_evidence,
                rotation_gate::from_quat(
                    input.match_input.held_affordance
                        .hand_in_object.rotation));
        const Transform release_hand = hand_world(
            release_sample.pose,
            input.match_input.held_affordance.hand);
        const rotation_gate::Rotation release_hand_rotation_evidence =
            hand_world_rotation_evidence(
                release_sample.pose,
                input.match_input.held_affordance.hand,
                release_probe.mapped_root_rotation_evidence());
        const vec3 release_translation =
            goal_hand.position - release_hand.position;
        const quat release_rotation = normalized(quat_mul(
            goal_hand.rotation,
            quat_inv(normalized(release_hand.rotation))));
        if (!finite(release_translation) || !valid_rotation(release_rotation) ||
            distance(goal_hand.position, release_hand.position) >
                ik_config_.maximum_request_position_m ||
            !rotation_gate::within(
                goal_hand_rotation_evidence,
                release_hand_rotation_evidence,
                ik_config_.maximum_request_orientation_radians)) {
            return {false, Reason::CorrectionLimit};
        }

        const Transform initial_object = hand_derived_object(
            input.match_input.current_pose,
            input.match_input.held_affordance);
        if (!valid_transform(initial_object)) {
            return {false, Reason::TargetUnavailable};
        }
        const double ticks = std::ceil(
            static_cast<double>(config_.timing.entry_blend_seconds) *
            kCanonicalFps);
        if (!(ticks >= 1.0) ||
            ticks > static_cast<double>(std::numeric_limits<int32_t>::max())) {
            return {false, Reason::CorrectionLimit};
        }

        PlaceStep initial{};
        initial.pose = input.match_input.current_pose;
        initial.object_world = initial_object;
        initial.phase = PlacePhase::Align;
        initial.source_frame = input.candidate.entry_frame;
        initial.source_frame_exact = input.candidate.entry_frame;
        copy_candidate_provenance(initial, input.candidate);
        initial.support_sweep_clear = true;

        player_ = std::move(player);
        begin_ = input;
        goal_object_ = goal_object;
        goal_hand_ = goal_hand;
        goal_object_rotation_evidence_ = goal_object_rotation_evidence;
        goal_hand_rotation_evidence_ = goal_hand_rotation_evidence;
        release_hand_translation_ = release_translation;
        release_hand_rotation_ = release_rotation;
        last_safe_pose_ = input.match_input.current_pose;
        last_safe_object_ = initial_object;
        previous_object_ = initial_object;
        frozen_object_ = Transform{};
        last_step_ = initial;
        entry_blend_ticks_ = static_cast<int32_t>(ticks);
        output_ticks_ = 0;
        recovering_ = false;
        release_pending_ = false;
        release_pulse_emitted_ = false;
        release_acknowledged_ = false;
        started_ = true;
        return {true, Reason::None};
    } catch (const std::invalid_argument&) {
        return {false, Reason::TargetChanged};
    } catch (const std::out_of_range&) {
        return {false, Reason::TargetChanged};
    }
}

PlaceStep PlaceController::update(float dt) {
    if (!finite(dt) || dt != kCanonicalDt) {
        throw std::invalid_argument(
            "place controller requires one exact 0.04 second tick");
    }
    if (!started_) throw std::logic_error("place controller is not started");
    if (recovering_) return last_step_;
    if (release_pending_) {
        PlaceStep pending = last_step_;
        pending.release_due = false;
        last_step_ = pending;
        return pending;
    }

    PlacePlayer trial = player_;
    trial.advance(dt);
    const PlaceSample sample = trial.sample();
    const rotation_gate::Rotation mapped_source_root_evidence =
        trial.mapped_root_rotation_evidence();
    const int32_t next_output_tick = output_ticks_ + 1;
    Pose pose = corrected_source_pose(
        sample.pose,
        begin_,
        trial.source_frame_exact(),
        next_output_tick,
        entry_blend_ticks_);
    const rotation_gate::Rotation root_rotation_evidence =
        corrected_root_rotation_evidence(
            pose,
            mapped_source_root_evidence,
            begin_,
            trial.source_frame_exact(),
            next_output_tick,
            entry_blend_ticks_);

    PlaceStep step{};
    step.pose = pose;
    step.phase = sample.phase;
    step.source_frame = sample.source_frame;
    step.source_frame_exact = trial.source_frame_exact();
    step.committed = sample.committed;
    step.retract_finished = trial.finished();
    copy_candidate_provenance(step, begin_.candidate);

    if (release_acknowledged_) {
        step.object_world = frozen_object_;
        step.applied_vertical_correction_m =
            last_step_.applied_vertical_correction_m;
        step.support_sweep_clear = true;
        player_ = std::move(trial);
        output_ticks_ = next_output_tick;
        last_step_ = step;
        return step;
    }

    const float source_motion_progress = source_progress(
        begin_.candidate, trial.source_frame_exact());
    const float root_weight = 1.0F - smoothstep(source_motion_progress);
    step.requested_root_correction_m = root_weight * std::hypot(
        begin_.candidate.entry_root_offset.x,
        begin_.candidate.entry_root_offset.z);
    step.requested_yaw_correction_radians = std::abs(
        root_weight * begin_.candidate.entry_yaw_offset);
    step.applied_root_correction_m = std::hypot(
        pose.positions[kRoot].x - sample.pose.positions[kRoot].x,
        pose.positions[kRoot].z - sample.pose.positions[kRoot].z);
    step.applied_yaw_correction_radians = rotation_gate::radians(
        rotation_gate::measure(
            pose.rotations[kRoot], sample.pose.rotations[kRoot]));
    const Transform current_hand = hand_world(
        pose, begin_.match_input.held_affordance.hand);
    const rotation_gate::Rotation current_hand_rotation_evidence =
        hand_world_rotation_evidence(
            pose,
            begin_.match_input.held_affordance.hand,
            root_rotation_evidence);
    const float hand_progress = replay_progress(
        begin_.candidate,
        trial.source_frame_exact(),
        sample.committed);
    const WeightedHandTarget target_hand = weighted_hand_target(
        current_hand,
        release_hand_translation_,
        release_hand_rotation_,
        hand_progress);
    if (trial.release_due() &&
        (distance(goal_hand_.position, current_hand.position) >
             ik_config_.maximum_request_position_m ||
         !rotation_gate::within(
             goal_hand_rotation_evidence_,
             current_hand_rotation_evidence,
             ik_config_.maximum_request_orientation_radians))) {
        recovering_ = true;
        last_step_ = recovery_step(
            last_safe_pose_,
            last_safe_object_,
            last_step_,
            Reason::CorrectionLimit);
        return last_step_;
    }
    const double requested_position = distance(
        target_hand.value.position, current_hand.position);
    const rotation_gate::Rotation target_hand_rotation_evidence =
        hand_progress >= 1.0F
            ? goal_hand_rotation_evidence_
            : rotation_gate::multiply(
                  rotation_gate::from_quat(target_hand.correction),
                  current_hand_rotation_evidence);
    const rotation_gate::Measure requested_orientation =
        rotation_gate::measure(
            target_hand_rotation_evidence,
            current_hand_rotation_evidence);
    step.requested_hand_correction_m =
        static_cast<float>(requested_position);
    step.requested_hand_orientation_radians =
        rotation_gate::radians(requested_orientation);
    if (requested_position > ik_config_.maximum_request_position_m ||
        !rotation_gate::within(
            requested_orientation,
            ik_config_.maximum_request_orientation_radians)) {
        recovering_ = true;
        last_step_ = recovery_step(
            last_safe_pose_,
            last_safe_object_,
            last_step_,
            Reason::CorrectionLimit);
        return last_step_;
    }

    const IKResult ik = solve_hand_ik_with_rotation_evidence(
        pose,
        begin_.match_input.held_affordance.hand,
        target_hand.value,
        target_hand_rotation_evidence,
        root_rotation_evidence,
        ik_config_);
    if (!ik.accepted) {
        recovering_ = true;
        last_step_ = recovery_step(
            last_safe_pose_,
            last_safe_object_,
            last_step_,
            ik.reason);
        return last_step_;
    }
    step.pose = pose;
    step.hand_position_error_m = ik.position_error_m;
    step.hand_orientation_error_radians = ik.orientation_error_radians;
    const Transform corrected_hand = hand_world(
        pose, begin_.match_input.held_affordance.hand);
    step.applied_vertical_correction_m =
        corrected_hand.position.y - current_hand.position.y;
    step.applied_hand_correction_m = static_cast<float>(distance(
        corrected_hand.position, current_hand.position));
    step.applied_hand_orientation_radians = rotation_gate::radians(
        rotation_gate::measure(
            corrected_hand.rotation, current_hand.rotation));
    const Transform object_world = hand_derived_object(
        pose, begin_.match_input.held_affordance);
    if (!valid_transform(object_world)) {
        recovering_ = true;
        last_step_ = recovery_step(
            last_safe_pose_,
            last_safe_object_,
            last_step_,
            Reason::CorrectionLimit);
        return last_step_;
    }
    step.object_world = object_world;

    if (trial.release_due()) {
        const double position_error = distance(
            object_world.position, goal_object_.position);
        const rotation_gate::Rotation object_rotation_evidence =
            rotation_gate::multiply(
                hand_world_rotation_evidence(
                    pose,
                    begin_.match_input.held_affordance.hand,
                    root_rotation_evidence),
                rotation_gate::inverse(rotation_gate::from_quat(
                    begin_.match_input.held_affordance
                        .hand_in_object.rotation)));
        const rotation_gate::Measure orientation_error =
            rotation_gate::measure(
                object_rotation_evidence,
                goal_object_rotation_evidence_);
        const bool orientation_within = rotation_gate::within(
            orientation_error,
            config_.release_orientation_radians);
        step.hand_position_error_m = static_cast<float>(position_error);
        step.hand_orientation_error_radians =
            rotation_gate::classified_radians(
                orientation_error,
                config_.release_orientation_radians);
        if (position_error > config_.release_position_m) {
            recovering_ = true;
            last_step_ = recovery_step(
                last_safe_pose_,
                last_safe_object_,
                last_step_,
                Reason::ReleasePosition);
            last_step_.hand_position_error_m = step.hand_position_error_m;
            last_step_.hand_orientation_error_radians =
                step.hand_orientation_error_radians;
            return last_step_;
        }
        if (!orientation_within) {
            recovering_ = true;
            last_step_ = recovery_step(
                last_safe_pose_,
                last_safe_object_,
                last_step_,
                Reason::ReleaseOrientation);
            last_step_.hand_position_error_m = step.hand_position_error_m;
            last_step_.hand_orientation_error_radians =
                step.hand_orientation_error_radians;
            return last_step_;
        }
        try {
            step.actual_fit = evaluate_actual_placement_fit(
                begin_.match_input.surface,
                begin_.match_input.place_affordance,
                object_world,
                begin_.match_input.held_object_bounds);
        } catch (const std::exception&) {
            step.actual_fit = PlacementFit{};
            step.actual_fit.reason = Reason::PlacementOutOfBounds;
        }
        if (!step.actual_fit.accepted) {
            recovering_ = true;
            last_step_ = recovery_step(
                last_safe_pose_,
                last_safe_object_,
                last_step_,
                step.actual_fit.reason == Reason::None
                    ? Reason::PlacementOutOfBounds
                    : step.actual_fit.reason);
            last_step_.hand_position_error_m = step.hand_position_error_m;
            last_step_.hand_orientation_error_radians =
                step.hand_orientation_error_radians;
            last_step_.actual_fit = step.actual_fit;
            return last_step_;
        }
        step.support_sweep_clear = true;
        step.release_due = !release_pulse_emitted_;
        player_ = std::move(trial);
        output_ticks_ = next_output_tick;
        previous_object_ = object_world;
        last_safe_pose_ = pose;
        last_safe_object_ = object_world;
        release_pending_ = true;
        release_pulse_emitted_ = true;
        last_step_ = step;
        return step;
    }

    const bool endpoint_blocked = place_collision::object_intersects_support(
        object_world,
        begin_.match_input.held_object_bounds,
        begin_.match_input.surface);
    const bool sweep_blocked = place_collision::swept_interval_intersects_support(
        previous_object_,
        object_world,
        begin_.match_input.held_object_bounds,
        begin_.match_input.surface);
    if (endpoint_blocked || sweep_blocked) {
        recovering_ = true;
        last_step_ = recovery_step(
            last_safe_pose_,
            last_safe_object_,
            last_step_,
            Reason::BlockedPath);
        return last_step_;
    }

    step.support_sweep_clear = true;
    player_ = std::move(trial);
    output_ticks_ = next_output_tick;
    previous_object_ = object_world;
    last_safe_pose_ = pose;
    last_safe_object_ = object_world;
    last_step_ = step;
    return step;
}

void PlaceController::acknowledge_release(Transform placed_world) {
    if (!started_) throw std::logic_error("place controller is not started");
    if (!release_pending_ || release_acknowledged_) {
        throw std::logic_error("place release is not pending");
    }
    if (!valid_transform(placed_world) ||
        !same_transform(placed_world, last_safe_object_)) {
        throw std::invalid_argument(
            "placed object must equal the pending hand-derived transform");
    }
    PlacePlayer trial = player_;
    trial.acknowledge_release();
    player_ = std::move(trial);
    frozen_object_ = placed_world;
    release_pending_ = false;
    release_acknowledged_ = true;
    last_step_.release_due = false;
    last_step_.requested_root_correction_m = 0.0F;
    last_step_.applied_root_correction_m = 0.0F;
    last_step_.requested_yaw_correction_radians = 0.0F;
    last_step_.applied_yaw_correction_radians = 0.0F;
    last_step_.requested_hand_correction_m = 0.0F;
    last_step_.applied_hand_correction_m = 0.0F;
    last_step_.requested_hand_orientation_radians = 0.0F;
    last_step_.applied_hand_orientation_radians = 0.0F;
}

PlaceStep PlaceController::cancel() {
    if (!started_) throw std::logic_error("place controller is not started");
    if (recovering_) return last_step_;
    if (player_.committed() || release_pending_ || release_acknowledged_) {
        PlaceStep ignored = last_step_;
        ignored.release_due = false;
        last_step_ = ignored;
        return ignored;
    }
    recovering_ = true;
    last_step_ = recovery_step(
        last_safe_pose_,
        last_safe_object_,
        last_step_,
        Reason::Cancelled);
    return last_step_;
}

}  // namespace interaction
