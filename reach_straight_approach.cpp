#include "reach_straight_approach.h"

// Straight-in grasp approach behavior is implemented test-first by the
// supervised feature job.

#include "g1_arm_joint_metadata.h"
#include "g1_skeleton.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <tuple>

namespace reach {

namespace {

bool finite_vec3(vec3 v) {
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

bool finite_quat(quat value) {
    return std::isfinite(value.w) && std::isfinite(value.x) &&
        std::isfinite(value.y) && std::isfinite(value.z) &&
        quat_length(value) > 1.0e-8F;
}

bool config_finite(const StraightApproachConfig& c) {
    bool finite_hand_targets = true;
    for (size_t dof = 0U; dof < c.open_active_hand_dof.size(); ++dof) {
        finite_hand_targets =
            finite_hand_targets &&
            std::isfinite(c.open_active_hand_dof[dof]) &&
            std::isfinite(c.closed_active_hand_dof[dof]);
    }
    return finite_hand_targets &&
        std::isfinite(c.corridor_length_m) && c.corridor_length_m > 0.0F &&
        std::isfinite(c.corridor_radius_m) && c.corridor_radius_m > 0.0F &&
        std::isfinite(c.close_distance_m) && c.close_distance_m >= 0.0F &&
        std::isfinite(c.maximum_backward_step_m) &&
        c.maximum_backward_step_m >= 0.0F &&
        std::isfinite(c.maximum_blend_seconds) &&
        c.maximum_blend_seconds >= 0.0F &&
        std::isfinite(c.maximum_pregrasp_orientation_radians) &&
        c.maximum_pregrasp_orientation_radians >= 0.0F &&
        std::isfinite(c.open_gripper_radius_m) &&
        c.open_gripper_radius_m > 0.0F &&
        std::isfinite(c.closed_gripper_radius_m) &&
        c.closed_gripper_radius_m > 0.0F &&
        c.closed_gripper_radius_m <= c.open_gripper_radius_m;
}

StraightApproachQuality invalid_quality() {
    StraightApproachQuality q{};
    q.finite = false;
    q.reaches_pregrasp_plane = false;
    const float inf = std::numeric_limits<float>::infinity();
    q.maximum_lateral_m = inf;
    q.rms_lateral_m = inf;
    q.backward_ratio = inf;
    q.maximum_angle_radians = inf;
    q.rms_angle_radians = inf;
    q.corridor_start = 0U;
    return q;
}

}  // namespace

bool valid_straight_approach_config(
    const StraightApproachConfig& config) {
    return config_finite(config);
}

StraightApproachQuality measure_straight_approach(
    const std::vector<vec3>& wrist_path,
    vec3 contact_world,
    vec3 approach_world,
    const StraightApproachConfig& config) {
    if (!config_finite(config) || wrist_path.empty() ||
        !finite_vec3(contact_world) || !finite_vec3(approach_world)) {
        return invalid_quality();
    }
    const float approach_length = length(approach_world);
    if (!std::isfinite(approach_length) ||
        std::fabs(approach_length - 1.0F) > 1.0e-3F) {
        return invalid_quality();
    }
    const vec3 approach = approach_world / approach_length;

    const size_t n = wrist_path.size();
    for (size_t i = 0U; i < n; ++i) {
        if (!finite_vec3(wrist_path[i])) return invalid_quality();
    }

    // Axial distance from each wrist sample back to contact along approach.
    std::vector<float> remaining(n, 0.0F);
    float max_remaining = -std::numeric_limits<float>::infinity();
    for (size_t i = 0U; i < n; ++i) {
        const vec3 from_contact = contact_world - wrist_path[i];
        remaining[i] = dot(from_contact, approach);
        if (remaining[i] > max_remaining) max_remaining = remaining[i];
    }

    // Earliest final-contiguous sample within the corridor length of contact.
    size_t corridor_start = n - 1U;
    const float corridor_edge = config.corridor_length_m + 1.0e-4F;
    for (size_t i = n; i-- > 0U;) {
        if (remaining[i] <= corridor_edge) {
            corridor_start = i;
        } else {
            break;
        }
    }

    StraightApproachQuality quality{};
    quality.finite = true;
    quality.corridor_start = corridor_start;
    quality.reaches_pregrasp_plane =
        max_remaining >= config.corridor_length_m - 1.0e-4F;

    // Lateral drift measured on the corridor suffix samples.
    float max_lateral = 0.0F;
    double sum_lateral_sq = 0.0;
    size_t lateral_count = 0U;
    for (size_t i = corridor_start; i < n; ++i) {
        const vec3 from_contact = contact_world - wrist_path[i];
        const vec3 lateral_vector = from_contact - remaining[i] * approach;
        const float lateral = length(lateral_vector);
        if (lateral > max_lateral) max_lateral = lateral;
        sum_lateral_sq += static_cast<double>(lateral) * lateral;
        ++lateral_count;
    }
    quality.maximum_lateral_m = max_lateral;
    quality.rms_lateral_m = lateral_count > 0U
        ? static_cast<float>(std::sqrt(sum_lateral_sq / lateral_count))
        : 0.0F;

    // Segment-wise backward motion and approach-angle error.
    float backward_sum = 0.0F;
    float axial_abs_sum = 0.0F;
    float max_angle = 0.0F;
    double sum_angle_sq = 0.0;
    size_t angle_count = 0U;
    for (size_t i = corridor_start + 1U; i < n; ++i) {
        const vec3 delta = wrist_path[i] - wrist_path[i - 1U];
        const float axial = dot(delta, approach);
        axial_abs_sum += std::fabs(axial);
        if (axial < 0.0F) backward_sum += -axial;
        const float seg_length = length(delta);
        if (seg_length > 1.0e-6F) {
            const float cos_angle = axial / seg_length;
            const float clamped =
                cos_angle > 1.0F ? 1.0F : (cos_angle < -1.0F ? -1.0F : cos_angle);
            const float angle = std::acos(clamped);
            if (angle > max_angle) max_angle = angle;
            sum_angle_sq += static_cast<double>(angle) * angle;
            ++angle_count;
        }
    }
    quality.backward_ratio =
        axial_abs_sum > 1.0e-9F ? backward_sum / axial_abs_sum : 0.0F;
    quality.maximum_angle_radians = max_angle;
    quality.rms_angle_radians = angle_count > 0U
        ? static_cast<float>(std::sqrt(sum_angle_sq / angle_count))
        : 0.0F;

    return quality;
}

bool straight_approach_quality_less(
    const StraightApproachQuality& left,
    const StraightApproachQuality& right) {
    const int left_missing = left.reaches_pregrasp_plane ? 0 : 1;
    const int right_missing = right.reaches_pregrasp_plane ? 0 : 1;
    return std::tie(
        left_missing,
        left.maximum_lateral_m,
        left.backward_ratio,
        left.rms_angle_radians,
        left.maximum_angle_radians) <
      std::tie(
        right_missing,
        right.maximum_lateral_m,
        right.backward_ratio,
        right.rms_angle_radians,
        right.maximum_angle_radians);
}

namespace {

float smoothstep(float value) {
    const float x = value < 0.0F ? 0.0F : (value > 1.0F ? 1.0F : value);
    return x * x * (3.0F - 2.0F * x);
}

float lerp_scalar(float a, float b, float t) {
    return a + (b - a) * t;
}

struct SweepInterval {
    bool intersects = false;
    float entry = 0.0F;
};

// Conservative swept-sphere versus oriented-box interval. Transforms the
// segment into the box frame and expands the box half-extents by the sphere
// radius.
SweepInterval swept_sphere_box_interval(
    vec3 start,
    vec3 stop,
    float radius,
    const interaction::OrientedBox& box) {
    const interaction::Transform box_frame{
        box.world.position, quat_normalize(box.world.rotation)};
    const vec3 local_start =
        interaction::compose(
            interaction::inverse(box_frame),
            interaction::Transform{start, quat()}).position;
    const vec3 local_stop =
        interaction::compose(
            interaction::inverse(box_frame),
            interaction::Transform{stop, quat()}).position;
    const vec3 delta = local_stop - local_start;
    const vec3 expanded = box.dimensions * 0.5F + radius;
    float minimum_time = 0.0F;
    float maximum_time = 1.0F;
    const float epsilon = 1.0e-8F;
    const float axis_start[3] = {local_start.x, local_start.y, local_start.z};
    const float axis_delta[3] = {delta.x, delta.y, delta.z};
    const float half[3] = {expanded.x, expanded.y, expanded.z};
    for (size_t axis = 0U; axis < 3U; ++axis) {
        if (std::fabs(axis_delta[axis]) <= epsilon) {
            if (axis_start[axis] < -half[axis] ||
                axis_start[axis] > half[axis]) {
                return {};
            }
        } else {
            float first = (-half[axis] - axis_start[axis]) / axis_delta[axis];
            float second = (half[axis] - axis_start[axis]) / axis_delta[axis];
            if (first > second) std::swap(first, second);
            minimum_time = std::max(minimum_time, first);
            maximum_time = std::min(maximum_time, second);
            if (minimum_time > maximum_time) return {};
        }
    }
    return {true, minimum_time};
}

float wrap_angle(float value) {
    const float pi = 3.14159265358979323846F;
    value = std::fmod(value + pi, 2.0F * pi);
    if (value < 0.0F) value += 2.0F * pi;
    return value - pi;
}

interaction::Hand interaction_hand(Hand hand) {
    return hand == Hand::Left ? interaction::Hand::Left
                              : interaction::Hand::Right;
}

size_t wrist_bone(Hand hand) {
    return hand == Hand::Left
        ? static_cast<size_t>(g1_skeleton::LeftWrist)
        : static_cast<size_t>(g1_skeleton::RightWrist);
}

size_t elbow_bone(Hand hand) {
    return hand == Hand::Left
        ? static_cast<size_t>(g1_skeleton::LeftElbow)
        : static_cast<size_t>(g1_skeleton::RightElbow);
}

// Base index into Pose::hand_dof for the active hand's seven DOFs.
size_t active_hand_dof_base(Hand hand) {
    return hand == Hand::Left ? 0U : 7U;
}

const interaction::HingeJoint& upper_body_metadata(Hand hand, size_t joint) {
    if (joint < interaction::kWaist.size()) {
        return interaction::kWaist[joint];
    }
    const size_t arm = joint - interaction::kWaist.size();
    return hand == Hand::Left ? interaction::kLeftArm[arm]
                              : interaction::kRightArm[arm];
}

bool finite_pose(const interaction::Pose& pose) {
    for (size_t bone = 0U; bone < g1_skeleton::BoneCount; ++bone) {
        const vec3 p = pose.positions[bone];
        const vec3 v = pose.velocities[bone];
        const quat r = pose.rotations[bone];
        const vec3 w = pose.angular_velocities[bone];
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z) ||
            !std::isfinite(v.x) || !std::isfinite(v.y) || !std::isfinite(v.z) ||
            !std::isfinite(r.w) || !std::isfinite(r.x) ||
            !std::isfinite(r.y) || !std::isfinite(r.z) ||
            !std::isfinite(w.x) || !std::isfinite(w.y) || !std::isfinite(w.z)) {
            return false;
        }
        if (quat_length(r) <= 1.0e-8F) return false;
    }
    for (size_t dof = 0U; dof < pose.hand_dof.size(); ++dof) {
        if (!std::isfinite(pose.hand_dof[dof]) ||
            !std::isfinite(pose.hand_dof_velocities[dof])) {
            return false;
        }
    }
    return true;
}

bool valid_box(const interaction::OrientedBox& box) {
    return finite_vec3(box.world.position) && finite_quat(box.world.rotation) &&
        finite_vec3(box.dimensions) &&
        box.dimensions.x > 0.0F && box.dimensions.y > 0.0F &&
        box.dimensions.z > 0.0F;
}

bool valid_collision_config(
    const interaction::TrajectoryCollisionConfig& config) {
    return std::isfinite(config.wrist_radius_m) &&
        std::isfinite(config.forearm_radius_m) &&
        std::isfinite(config.joint_radius_m) &&
        std::isfinite(config.limb_radius_m) &&
        std::isfinite(config.torso_radius_m) &&
        config.wrist_radius_m > 0.0F &&
        config.forearm_radius_m > 0.0F &&
        config.joint_radius_m > 0.0F &&
        config.limb_radius_m > 0.0F &&
        config.torso_radius_m > 0.0F;
}

float rotation_error(quat current, quat target) {
    const quat delta = quat_abs(quat_mul(
        quat_normalize(current), quat_inv(quat_normalize(target))));
    return length(quat_to_scaled_angle_axis(delta));
}

vec3 active_wrist_world(const interaction::Pose& pose, Hand hand) {
    return interaction::world_pose(pose).positions[wrist_bone(hand)];
}

CorridorRetargetResult failed(CorridorRetargetFailure failure, size_t sample) {
    CorridorRetargetResult result{};
    result.accepted = false;
    result.failure = failure;
    result.failure_sample = sample;
    return result;
}

CorridorRetargetResult failed(
    CorridorRetargetResult result,
    CorridorRetargetFailure failure,
    size_t sample) {
    result.accepted = false;
    result.failure = failure;
    result.failure_sample = sample;
    return result;
}

}  // namespace

CorridorRetargetResult retarget_straight_approach(
    const std::vector<interaction::Pose>& source,
    Hand hand,
    const interaction::Transform& hand_world,
    vec3 approach_world,
    const interaction::OrientedBox& object,
    const interaction::EnvironmentGeometry& environment,
    float fps,
    const StraightApproachConfig& config,
    const interaction::PostureIKConfig& ik_config,
    const interaction::TrajectoryCollisionConfig& collision_config) {
    if (source.size() < 2U || !config_finite(config) ||
        !std::isfinite(fps) || fps <= 0.0F ||
        !finite_vec3(hand_world.position) || !finite_quat(hand_world.rotation) ||
        !valid_box(object) || !valid_collision_config(collision_config)) {
        return failed(CorridorRetargetFailure::InvalidInput, 0U);
    }
    for (const interaction::OrientedBox& box : environment.boxes) {
        if (!valid_box(box)) {
            return failed(CorridorRetargetFailure::InvalidInput, 0U);
        }
    }
    const float approach_length = length(approach_world);
    if (!std::isfinite(approach_length) ||
        std::fabs(approach_length - 1.0F) > 1.0e-3F) {
        return failed(CorridorRetargetFailure::InvalidInput, 0U);
    }
    const vec3 approach = approach_world / approach_length;
    for (const interaction::Pose& pose : source) {
        if (!finite_pose(pose)) {
            return failed(CorridorRetargetFailure::InvalidInput, 0U);
        }
    }

    // Corridor coverage of the source path relative to the requested grasp.
    std::vector<vec3> source_path(source.size());
    for (size_t i = 0U; i < source.size(); ++i) {
        source_path[i] = active_wrist_world(source[i], hand);
    }
    const StraightApproachQuality quality = measure_straight_approach(
        source_path, hand_world.position, approach, config);
    if (!quality.finite) {
        return failed(CorridorRetargetFailure::InvalidInput, 0U);
    }
    if (!quality.reaches_pregrasp_plane) {
        CorridorRetargetResult result =
            failed(CorridorRetargetFailure::NoPregraspCoverage, 0U);
        result.quality = quality;
        return result;
    }

    const size_t corridor_start = quality.corridor_start;
    const double requested_blend_samples = std::ceil(
        static_cast<double>(config.maximum_blend_seconds) *
        static_cast<double>(fps));
    const size_t maximum_blend_samples = static_cast<size_t>(
        std::min(
            requested_blend_samples,
            static_cast<double>(source.size())));
    const size_t blend_start = corridor_start > maximum_blend_samples
        ? corridor_start - maximum_blend_samples
        : 0U;

    const vec3 pregrasp =
        hand_world.position - config.corridor_length_m * approach;

    CorridorRetargetResult result{};
    result.blend_start = blend_start;
    result.corridor_start = corridor_start;
    result.quality = quality;
    result.poses = source;  // Pre-blend samples remain bit-identical.

    const interaction::Hand ihand = interaction_hand(hand);
    const size_t dof_base = active_hand_dof_base(hand);
    const size_t upper_body_joint_count = interaction::kUpperBodyJointCount;
    std::array<float, 7U> closed_hand_dof =
        config.closed_active_hand_dof;
    if (!config.use_explicit_closed_hand_dof) {
        for (size_t dof = 0U; dof < closed_hand_dof.size(); ++dof) {
            closed_hand_dof[dof] =
                source.back().hand_dof[dof_base + dof];
        }
    }
    bool has_hand_closure = false;
    for (size_t dof = 0U; dof < closed_hand_dof.size(); ++dof) {
        has_hand_closure =
            has_hand_closure ||
            std::fabs(
                closed_hand_dof[dof] -
                config.open_active_hand_dof[dof]) > 1.0e-5F;
    }
    std::vector<float> gripper_close_weights(source.size(), 0.0F);

    interaction::UpperBodyAngles previous_source{};
    interaction::UpperBodyAngles previous_solution{};
    bool have_previous = false;

    // Monotonic axial progress: clamps backward steps in the source.
    float previous_remaining = config.corridor_length_m;

    float deformation_sum = 0.0F;
    size_t deformation_samples = 0U;

    for (size_t sample = blend_start; sample < source.size(); ++sample) {
        interaction::Pose pose = source[sample];  // Preserves root and legs.
        const interaction::Pose& placed = source[sample];

        interaction::Transform desired{};
        if (sample < corridor_start) {
            // Blend from the recorded wrist to the exact pre-grasp target.
            const float u = static_cast<float>(sample - blend_start) /
                static_cast<float>(corridor_start - blend_start);
            const float weight = smoothstep(u);
            const vec3 recorded = source_path[sample];
            desired.position = lerp(recorded, pregrasp, weight);
            desired.rotation = quat_nlerp_shortest(
                interaction::world_pose(placed)
                    .rotations[wrist_bone(hand)],
                hand_world.rotation,
                weight);
        } else {
            // Inside the corridor: monotonic progress toward contact.
            float remaining = config.corridor_length_m;
            if (sample > corridor_start) {
                const vec3 from_contact =
                    hand_world.position - source_path[sample];
                remaining = dot(from_contact, approach);
                remaining = std::min(remaining, previous_remaining);
                remaining = std::max(remaining, 0.0F);
            }
            previous_remaining = remaining;
            const float progress =
                1.0F - remaining / config.corridor_length_m;
            desired.position =
                lerp(pregrasp, hand_world.position, smoothstep(progress));
            desired.rotation = hand_world.rotation;
        }
        if (sample + 1U == source.size()) {
            // The terminal frame is the grasp constraint, independent of any
            // residual axial error in the recorded candidate.
            desired = hand_world;
        }

        const interaction::UpperBodyAngles source_angles =
            interaction::decompose_upper_body(placed, ihand);
        interaction::UpperBodyAngles temporal_seed = source_angles;
        if (have_previous) {
            for (size_t joint = 0U; joint < upper_body_joint_count; ++joint) {
                temporal_seed[joint] += wrap_angle(
                    previous_solution[joint] - previous_source[joint]);
            }
        }

        const interaction::PostureIKResult ik =
            interaction::solve_hand_posture_ik_task_priority(
                pose, ihand, desired, placed, temporal_seed, ik_config);
        previous_source = source_angles;
        previous_solution = ik.joint_angles;
        have_previous = true;

        if (!ik.accepted || !finite_pose(pose) ||
            !std::isfinite(ik.position_error_m) ||
            !std::isfinite(ik.orientation_error_radians) ||
            !std::isfinite(static_cast<float>(ik.objective))) {
            return failed(
                std::move(result),
                CorridorRetargetFailure::InvalidSolver,
                sample);
        }

        for (size_t joint = 0U; joint < upper_body_joint_count; ++joint) {
            const interaction::HingeJoint& metadata =
                upper_body_metadata(hand, joint);
            const size_t bone = static_cast<size_t>(metadata.bone);
            const quat delta = quat_abs(quat_mul(
                quat_normalize(placed.rotations[bone]),
                quat_inv(quat_normalize(pose.rotations[bone]))));
            const float angle = length(quat_to_scaled_angle_axis(delta));
            deformation_sum += angle * angle;
        }
        ++deformation_samples;

        // Gripper schedule: hold open until close_distance_m remains.
        const vec3 wrist_now = active_wrist_world(pose, hand);
        const float remaining_to_contact =
            dot(hand_world.position - wrist_now, approach);
        float close_u = 0.0F;
        if (config.close_distance_m > 1.0e-6F) {
            close_u = (config.close_distance_m - remaining_to_contact) /
                config.close_distance_m;
        } else {
            close_u = remaining_to_contact <= 0.0F ? 1.0F : 0.0F;
        }
        const float close_weight = smoothstep(close_u);
        gripper_close_weights[sample] =
            has_hand_closure ? close_weight : 0.0F;
        for (size_t dof = 0U; dof < 7U; ++dof) {
            const float open = config.open_active_hand_dof[dof];
            const float closed = closed_hand_dof[dof];
            pose.hand_dof[dof_base + dof] =
                lerp_scalar(open, closed, close_weight);
        }

        if (!finite_pose(pose)) {
            return failed(
                std::move(result),
                CorridorRetargetFailure::InvalidSolver,
                sample);
        }
        result.poses[sample] = std::move(pose);
    }

    // Active-hand DOF velocities recomputed at the native fps.
    for (size_t sample = blend_start; sample < result.poses.size(); ++sample) {
        for (size_t dof = 0U; dof < 7U; ++dof) {
            const size_t index = dof_base + dof;
            if (sample == 0U) {
                result.poses[sample].hand_dof_velocities[index] = 0.0F;
            } else {
                result.poses[sample].hand_dof_velocities[index] =
                    (result.poses[sample].hand_dof[index] -
                     result.poses[sample - 1U].hand_dof[index]) * fps;
            }
        }
        if (!finite_pose(result.poses[sample])) {
            return failed(
                std::move(result),
                CorridorRetargetFailure::InvalidSolver,
                sample);
        }
    }

    result.active_arm_deformation = deformation_samples > 0U
        ? deformation_sum /
            static_cast<float>(deformation_samples * upper_body_joint_count)
        : 0.0F;

    // Corridor and monotonicity validation on the solved wrist path.
    float prev_remaining = std::numeric_limits<float>::infinity();
    for (size_t sample = corridor_start; sample < result.poses.size();
         ++sample) {
        const vec3 wrist = active_wrist_world(result.poses[sample], hand);
        const vec3 from_contact = hand_world.position - wrist;
        const float remaining = dot(from_contact, approach);
        const vec3 lateral = from_contact - remaining * approach;
        if (length(lateral) > config.corridor_radius_m + 1.0e-4F) {
            return failed(
                std::move(result),
                CorridorRetargetFailure::OutsideCorridor,
                sample);
        }
        if (sample > corridor_start) {
            const float step = prev_remaining - remaining;
            if (step < -config.maximum_backward_step_m - 1.0e-5F) {
                return failed(
                    std::move(result),
                    CorridorRetargetFailure::BackwardMotion,
                    sample);
            }
        }
        prev_remaining = remaining;
    }

    const interaction::WorldPose pregrasp_world =
        interaction::world_pose(result.poses[corridor_start]);
    if (length(
            pregrasp_world.positions[wrist_bone(hand)] - pregrasp) >
            0.002F ||
        rotation_error(
            pregrasp_world.rotations[wrist_bone(hand)],
            hand_world.rotation) >
            config.maximum_pregrasp_orientation_radians + 1.0e-5F) {
        return failed(
            std::move(result),
            CorridorRetargetFailure::InvalidSolver,
            corridor_start);
    }
    const interaction::WorldPose contact_world =
        interaction::world_pose(result.poses.back());
    const size_t contact_sample = result.poses.size() - 1U;
    if (length(
            contact_world.positions[wrist_bone(hand)] -
            hand_world.position) > 0.002F ||
        rotation_error(
            contact_world.rotations[wrist_bone(hand)],
            hand_world.rotation) >
            std::min(
                ik_config.accepted_orientation_radians,
                config.maximum_pregrasp_orientation_radians) + 1.0e-5F) {
        return failed(
            std::move(result),
            CorridorRetargetFailure::InvalidSolver,
            contact_sample);
    }

    // Reject abrupt changes in the retarget correction itself. Recorded joint
    // motion remains allowed; only an inter-frame jump introduced by IK is
    // bounded. The 0.35 radian floor accommodates the solver redistributing
    // correction between waist and arm while still rejecting visible flips.
    const float maximum_correction_step =
        std::max(0.35F, 2.0F * ik_config.maximum_step_radians) + 1.0e-5F;
    for (size_t sample = blend_start + 1U; sample < result.poses.size();
         ++sample) {
        const interaction::UpperBodyAngles source_previous =
            interaction::decompose_upper_body(source[sample - 1U], ihand);
        const interaction::UpperBodyAngles source_current =
            interaction::decompose_upper_body(source[sample], ihand);
        const interaction::UpperBodyAngles solved_previous =
            interaction::decompose_upper_body(result.poses[sample - 1U], ihand);
        const interaction::UpperBodyAngles solved_current =
            interaction::decompose_upper_body(result.poses[sample], ihand);
        for (size_t joint = 0U; joint < upper_body_joint_count; ++joint) {
            const float previous_correction = wrap_angle(
                solved_previous[joint] - source_previous[joint]);
            const float current_correction = wrap_angle(
                solved_current[joint] - source_current[joint]);
            if (std::fabs(wrap_angle(
                    current_correction - previous_correction)) >
                maximum_correction_step) {
                return failed(
                    std::move(result),
                    CorridorRetargetFailure::InvalidSolver,
                    sample);
            }
        }
    }

    // Dynamic open-gripper proxy: swept wrist sphere against object/environment.
    constexpr float terminal_contact_time_epsilon = 1.0e-5F;
    for (size_t sample = blend_start; sample < result.poses.size(); ++sample) {
        const size_t previous_sample =
            sample == blend_start ? sample : sample - 1U;
        const vec3 previous =
            active_wrist_world(result.poses[previous_sample], hand);
        const vec3 current = active_wrist_world(result.poses[sample], hand);
        const float conservative_close_weight = std::min(
            gripper_close_weights[previous_sample],
            gripper_close_weights[sample]);
        const float radius = lerp_scalar(
            config.open_gripper_radius_m,
            config.closed_gripper_radius_m,
            conservative_close_weight);
        const bool terminal = sample == result.poses.size() - 1U;
        const SweepInterval object_interval =
            swept_sphere_box_interval(previous, current, radius, object);
        const bool endpoint_only_contact =
            object_interval.entry >=
            1.0F - terminal_contact_time_epsilon;
        if (object_interval.intersects &&
            !(terminal && endpoint_only_contact)) {
            return failed(
                std::move(result),
                CorridorRetargetFailure::ObjectCollision,
                sample);
        }
        for (const interaction::OrientedBox& box : environment.boxes) {
            if (swept_sphere_box_interval(
                    previous, current, radius, box).intersects) {
                return failed(
                    std::move(result),
                    CorridorRetargetFailure::EnvironmentCollision, sample);
            }
        }
    }

    // Full-body revalidation with the existing trajectory collision oracle.
    interaction::ShapedHandTrajectory shaped{};
    shaped.poses = result.poses;
    shaped.contact_accepted = true;
    shaped.path.hands.reserve(result.poses.size());
    shaped.path.elbows.reserve(result.poses.size());
    for (const interaction::Pose& pose : result.poses) {
        const interaction::WorldPose world = interaction::world_pose(pose);
        shaped.path.hands.push_back(
            {world.positions[wrist_bone(hand)],
             world.rotations[wrist_bone(hand)]});
        shaped.path.elbows.push_back(world.positions[elbow_bone(hand)]);
    }
    const interaction::TrajectoryFeasibility feasibility =
        interaction::evaluate_shaped_trajectory_feasibility(
            shaped, result.poses.size() - 1U, ihand, object, environment,
            collision_config);
    if (feasibility.reason ==
        interaction::TrajectoryFeasibilityReason::ObjectCollision) {
        return failed(
            std::move(result),
            CorridorRetargetFailure::ObjectCollision,
            feasibility.sample);
    }
    if (feasibility.reason ==
        interaction::TrajectoryFeasibilityReason::EnvironmentCollision) {
        return failed(
            std::move(result),
            CorridorRetargetFailure::EnvironmentCollision,
            feasibility.sample);
    }

    result.accepted = true;
    result.failure = CorridorRetargetFailure::None;
    return result;
}

}  // namespace reach
