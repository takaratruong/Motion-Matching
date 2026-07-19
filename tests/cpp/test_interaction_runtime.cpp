#include "interaction_runtime.h"
#include "g1_arm_joint_metadata.h"
#include "tests/cpp/interaction_runtime_fixture.h"

#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace interaction {

struct PickBuildObservation {
    bool accepted = false;
    Reason reason = Reason::None;
};

struct InteractionRuntimeTestAccess {
    static PickBuildObservation build_pick_evaluation(
        const InteractionRuntime& runtime,
        const LocomotionSnapshot& locomotion,
        TargetHandle target,
        uint32_t affordance_id,
        bool require_free) {
        const InteractionRuntime::PickEvaluationBuild built =
            runtime.build_pick_evaluation(
                locomotion, target, affordance_id, require_free);
        return {built.accepted, built.reason};
    }

    static const std::optional<MatchCandidate>& candidate(
        const InteractionRuntime& runtime) {
        return runtime.candidate_;
    }

    static std::optional<PlaceMatchInput> place_match_input(
        const InteractionRuntime& runtime,
        SurfaceHandle surface,
        uint32_t affordance_id) {
        const InteractionRuntime::PlaceMatchBuildResult built =
            runtime.make_place_match_input(surface, affordance_id);
        if (!built.accepted) return std::nullopt;
        return built.input;
    }

    static float player_elapsed_seconds(
        const InteractionRuntime& runtime) {
        return runtime.player_.value().elapsed_seconds();
    }

    static int32_t player_frame(const InteractionRuntime& runtime) {
        return runtime.player_.value().frame();
    }

    static float clearance_elapsed_seconds(
        const InteractionRuntime& runtime) {
        return runtime.clearance_player_.value().elapsed_seconds();
    }

    static int32_t clearance_frame(const InteractionRuntime& runtime) {
        return runtime.clearance_player_.value().frame();
    }
};

}  // namespace interaction

namespace {

constexpr float kDt = 1.0F / 25.0F;
constexpr int kMaximumUpdates = 600;

interaction::RuntimeFixture make_runtime_fixture() {
    return interaction::make_place_runtime_fixture();
}

bool exact(vec3 left, vec3 right) {
    return left.x == right.x && left.y == right.y && left.z == right.z;
}

bool exact(quat left, quat right) {
    return left.w == right.w && left.x == right.x &&
           left.y == right.y && left.z == right.z;
}

bool exact(
    const interaction::Transform& left,
    const interaction::Transform& right) {
    return exact(left.position, right.position) &&
           exact(left.rotation, right.rotation);
}

bool exact(
    const interaction::Pose& left,
    const interaction::Pose& right) {
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

bool exact(
    const interaction::LocomotionSnapshot& left,
    const interaction::LocomotionSnapshot& right) {
    if (!exact(left.pose, right.pose)) return false;
    for (size_t index = 0;
         index < left.future_root_positions.size();
         ++index) {
        if (!exact(
                left.future_root_positions[index],
                right.future_root_positions[index]) ||
            !exact(
                left.future_root_rotations[index],
                right.future_root_rotations[index])) {
            return false;
        }
    }
    return true;
}

bool same_float_bits(float left, float right) {
    uint32_t left_bits = 0U;
    uint32_t right_bits = 0U;
    static_assert(sizeof(left_bits) == sizeof(left));
    std::memcpy(&left_bits, &left, sizeof(left_bits));
    std::memcpy(&right_bits, &right, sizeof(right_bits));
    return left_bits == right_bits;
}

bool same_bits(vec3 left, vec3 right) {
    return same_float_bits(left.x, right.x) &&
           same_float_bits(left.y, right.y) &&
           same_float_bits(left.z, right.z);
}

bool same_bits(quat left, quat right) {
    return same_float_bits(left.w, right.w) &&
           same_float_bits(left.x, right.x) &&
           same_float_bits(left.y, right.y) &&
           same_float_bits(left.z, right.z);
}

bool same_bits(
    const interaction::LocomotionSnapshot& left,
    const interaction::LocomotionSnapshot& right) {
    for (size_t bone = 0; bone < g1_skeleton::BoneCount; ++bone) {
        if (!same_bits(
                left.pose.positions[bone], right.pose.positions[bone]) ||
            !same_bits(
                left.pose.velocities[bone], right.pose.velocities[bone]) ||
            !same_bits(
                left.pose.rotations[bone], right.pose.rotations[bone]) ||
            !same_bits(
                left.pose.angular_velocities[bone],
                right.pose.angular_velocities[bone])) {
            return false;
        }
    }
    for (size_t index = 0; index < left.pose.hand_dof.size(); ++index) {
        if (!same_float_bits(
                left.pose.hand_dof[index], right.pose.hand_dof[index]) ||
            !same_float_bits(
                left.pose.hand_dof_velocities[index],
                right.pose.hand_dof_velocities[index])) {
            return false;
        }
    }
    if (left.pose.foot_contacts != right.pose.foot_contacts) return false;
    for (size_t index = 0;
         index < left.future_root_positions.size();
         ++index) {
        if (!same_bits(
                left.future_root_positions[index],
                right.future_root_positions[index]) ||
            !same_bits(
                left.future_root_rotations[index],
                right.future_root_rotations[index])) {
            return false;
        }
    }
    return true;
}

bool near(float left, float right, float tolerance = 2.0e-4F) {
    return std::abs(left - right) <= tolerance;
}

bool near(vec3 left, vec3 right, float tolerance = 2.0e-4F) {
    return near(left.x, right.x, tolerance) &&
           near(left.y, right.y, tolerance) &&
           near(left.z, right.z, tolerance);
}

bool near_rotation(quat left, quat right, float tolerance = 2.0e-4F) {
    const auto same_sign = [&]() {
        return near(left.w, right.w, tolerance) &&
               near(left.x, right.x, tolerance) &&
               near(left.y, right.y, tolerance) &&
               near(left.z, right.z, tolerance);
    };
    if (same_sign()) return true;
    right = -right;
    return same_sign();
}

float orientation_yaw(quat value) {
    value = value / quat_length(value);
    const vec3 forward = quat_mul_vec3(
        value, vec3(0.0F, 0.0F, 1.0F));
    return std::atan2(forward.x, forward.z);
}

std::vector<float*> snapshot_float_fields(
    interaction::LocomotionSnapshot& snapshot) {
    std::vector<float*> fields;
    const auto append_vec3 = [&](vec3& value) {
        fields.push_back(&value.x);
        fields.push_back(&value.y);
        fields.push_back(&value.z);
    };
    const auto append_quat = [&](quat& value) {
        fields.push_back(&value.w);
        fields.push_back(&value.x);
        fields.push_back(&value.y);
        fields.push_back(&value.z);
    };
    for (vec3& value : snapshot.pose.positions) append_vec3(value);
    for (vec3& value : snapshot.pose.velocities) append_vec3(value);
    for (quat& value : snapshot.pose.rotations) append_quat(value);
    for (vec3& value : snapshot.pose.angular_velocities) append_vec3(value);
    for (float& value : snapshot.pose.hand_dof) fields.push_back(&value);
    for (float& value : snapshot.pose.hand_dof_velocities) {
        fields.push_back(&value);
    }
    for (vec3& value : snapshot.future_root_positions) append_vec3(value);
    for (quat& value : snapshot.future_root_rotations) append_quat(value);
    return fields;
}

uint32_t canonical_float_bits(float value) {
    uint32_t bits = 0U;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    return (bits & 0x7fffffffU) == 0U ? 0U : bits;
}

float float_from_bits(uint32_t bits) {
    float value = 0.0F;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

void reference_hash_byte(uint64_t& digest, uint8_t value) {
    digest ^= value;
    digest *= 1099511628211ULL;
}

void reference_hash_float(uint64_t& digest, float value) {
    const uint32_t bits = canonical_float_bits(value);
    for (unsigned shift = 0U; shift < 32U; shift += 8U) {
        reference_hash_byte(
            digest, static_cast<uint8_t>((bits >> shift) & 0xffU));
    }
}

void reference_hash_vec3(uint64_t& digest, vec3 value) {
    reference_hash_float(digest, value.x);
    reference_hash_float(digest, value.y);
    reference_hash_float(digest, value.z);
}

void reference_hash_quat(uint64_t& digest, quat value) {
    const std::array<float, 4> components{
        value.w, value.x, value.y, value.z};
    bool negate = false;
    for (float component : components) {
        const uint32_t bits = canonical_float_bits(component);
        if ((bits & 0x7fffffffU) != 0U) {
            negate = (bits & 0x80000000U) != 0U;
            break;
        }
    }
    for (float component : components) {
        reference_hash_float(digest, negate ? -component : component);
    }
}

uint64_t reference_snapshot_fingerprint(
    const interaction::LocomotionSnapshot& snapshot) {
    uint64_t digest = 14695981039346656037ULL;
    for (vec3 value : snapshot.pose.positions) {
        reference_hash_vec3(digest, value);
    }
    for (vec3 value : snapshot.pose.velocities) {
        reference_hash_vec3(digest, value);
    }
    for (quat value : snapshot.pose.rotations) {
        reference_hash_quat(digest, value);
    }
    for (vec3 value : snapshot.pose.angular_velocities) {
        reference_hash_vec3(digest, value);
    }
    for (float value : snapshot.pose.hand_dof) {
        reference_hash_float(digest, value);
    }
    for (float value : snapshot.pose.hand_dof_velocities) {
        reference_hash_float(digest, value);
    }
    for (uint8_t value : snapshot.pose.foot_contacts) {
        reference_hash_byte(digest, value);
    }
    for (vec3 value : snapshot.future_root_positions) {
        reference_hash_vec3(digest, value);
    }
    for (quat value : snapshot.future_root_rotations) {
        reference_hash_quat(digest, value);
    }
    return digest == 0U ? 14695981039346656037ULL : digest;
}

bool exact(
    const interaction::GraspAffordance& left,
    const interaction::GraspAffordance& right) {
    return left.id == right.id && left.hand == right.hand &&
           exact(left.hand_in_object, right.hand_in_object) &&
           exact(
               left.approach_direction_object,
               right.approach_direction_object) &&
           left.clearance_radius == right.clearance_radius;
}

bool exact(
    const interaction::InteractionTarget& left,
    const interaction::InteractionTarget& right) {
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

bool exact(
    const interaction::PlaceAffordance& left,
    const interaction::PlaceAffordance& right) {
    return left.id == right.id &&
           exact(left.object_in_surface, right.object_in_surface) &&
           exact(left.support_point_object, right.support_point_object) &&
           exact(
               left.approach_direction_surface,
               right.approach_direction_surface) &&
           left.clearance_radius == right.clearance_radius;
}

bool exact(
    const interaction::PlacementSurface& left,
    const interaction::PlacementSurface& right) {
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

bool exact(
    const interaction::IKConfig& left,
    const interaction::IKConfig& right) {
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

bool exact(
    const interaction::PlaceCandidate& left,
    const interaction::PlaceCandidate& right) {
    return left.mode == right.mode && left.source_id == right.source_id &&
           left.selection_id == right.selection_id &&
           left.source_support_height_m == right.source_support_height_m &&
           left.requested_support_height_m ==
               right.requested_support_height_m &&
           left.target_support_height_m == right.target_support_height_m &&
           left.requested_vertical_correction_m ==
               right.requested_vertical_correction_m &&
           left.timing.canonical_fps == right.timing.canonical_fps &&
           left.timing.playback_speed == right.timing.playback_speed &&
           left.timing.entry_blend_seconds ==
               right.timing.entry_blend_seconds &&
           left.timing.reversed_commit_seconds ==
               right.timing.reversed_commit_seconds &&
           left.timing.maximum_alignment_seconds ==
               right.timing.maximum_alignment_seconds &&
           left.match.maximum_entry_root_error_m ==
               right.match.maximum_entry_root_error_m &&
           left.match.maximum_entry_yaw_error_radians ==
               right.match.maximum_entry_yaw_error_radians &&
           exact(left.ik, right.ik) && left.clip == right.clip &&
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

bool exact(
    const interaction::PlaceStagingPreview& left,
    const interaction::PlaceStagingPreview& right) {
    return left.accepted == right.accepted && left.ready == right.ready &&
           left.reason == right.reason &&
           exact(left.candidate, right.candidate) &&
           exact(left.ik, right.ik) &&
           left.ik_config_fingerprint == right.ik_config_fingerprint &&
           exact(left.staging_root_world, right.staging_root_world) &&
           left.root_error_m == right.root_error_m &&
           left.yaw_error_radians == right.yaw_error_radians;
}

bool exact(
    const interaction::PlacementFit& left,
    const interaction::PlacementFit& right) {
    return left.accepted == right.accepted && left.reason == right.reason &&
           left.support_gap_m == right.support_gap_m &&
           left.lowest_corner_m == right.lowest_corner_m &&
           left.highest_corner_m == right.highest_corner_m &&
           left.footprint_valid == right.footprint_valid &&
           left.overhead_valid == right.overhead_valid;
}

bool exact(
    const interaction::RuntimePlaceDiagnostics& left,
    const interaction::RuntimePlaceDiagnostics& right) {
    return left.surface == right.surface &&
           left.affordance_id == right.affordance_id &&
           left.mode == right.mode &&
           left.source_id == right.source_id &&
           left.selection_id == right.selection_id &&
           left.source_support_height_m == right.source_support_height_m &&
           left.requested_support_height_m ==
               right.requested_support_height_m &&
           left.target_support_height_m == right.target_support_height_m &&
           left.requested_vertical_correction_m ==
               right.requested_vertical_correction_m &&
           left.applied_vertical_correction_m ==
               right.applied_vertical_correction_m &&
           left.preview_available == right.preview_available &&
           exact(left.preview, right.preview) &&
           left.candidate_certified == right.candidate_certified &&
           left.preflight_config_identity ==
               right.preflight_config_identity &&
           exact(left.effective_ik, right.effective_ik) &&
           left.ik_config_fingerprint == right.ik_config_fingerprint &&
           exact(left.requested_goal_world, right.requested_goal_world) &&
           exact(left.requested_fit, right.requested_fit) &&
           exact(left.actual_fit, right.actual_fit) &&
           left.clip == right.clip &&
           left.source_frame == right.source_frame &&
           left.source_frame_exact == right.source_frame_exact &&
           left.commit_frame == right.commit_frame &&
           left.phase == right.phase &&
           left.time_to_release_seconds == right.time_to_release_seconds &&
           left.committed == right.committed &&
           left.release_due == right.release_due &&
           left.released == right.released &&
           left.support_sweep_clear == right.support_sweep_clear &&
           left.support_position_error_m == right.support_position_error_m &&
           left.support_orientation_error_radians ==
               right.support_orientation_error_radians &&
           left.requested_root_correction_m ==
               right.requested_root_correction_m &&
           left.applied_root_correction_m ==
               right.applied_root_correction_m &&
           left.requested_yaw_correction_radians ==
               right.requested_yaw_correction_radians &&
           left.applied_yaw_correction_radians ==
               right.applied_yaw_correction_radians &&
           left.requested_hand_correction_m ==
               right.requested_hand_correction_m &&
           left.applied_hand_correction_m ==
               right.applied_hand_correction_m &&
           left.requested_hand_orientation_radians ==
               right.requested_hand_orientation_radians &&
           left.applied_hand_orientation_radians ==
               right.applied_hand_orientation_radians &&
           left.reason == right.reason;
}

bool exact(
    const interaction::PickupSourceProvenance& left,
    const interaction::PickupSourceProvenance& right) {
    return left.clip_ordinal == right.clip_ordinal &&
           left.range_start == right.range_start &&
           left.range_stop == right.range_stop &&
           left.entry_global_frame == right.entry_global_frame &&
           left.contact_global_frame == right.contact_global_frame &&
           left.lift_global_frame == right.lift_global_frame &&
           left.hold_global_frame == right.hold_global_frame &&
           left.active_hand == right.active_hand &&
           left.object_profile_id == right.object_profile_id &&
           same_bits(
               left.object_bounds.center_object,
               right.object_bounds.center_object) &&
           same_bits(
               left.object_bounds.half_extents_object,
               right.object_bounds.half_extents_object) &&
           same_bits(
               left.hand_in_object.position,
               right.hand_in_object.position) &&
           same_bits(
               left.hand_in_object.rotation,
               right.hand_in_object.rotation) &&
           same_float_bits(
               left.source_support_height_m,
               right.source_support_height_m);
}

bool exact(
    const interaction::CertifiedPickupSourceIdentity& left,
    const interaction::CertifiedPickupSourceIdentity& right) {
    return exact(left.provenance, right.provenance) &&
           left.sequence_id == right.sequence_id &&
           left.object_id == right.object_id &&
           left.reverse_start_global_frame ==
               right.reverse_start_global_frame &&
           left.join_key_sha256 == right.join_key_sha256 &&
           left.source_id == right.source_id;
}

bool exact(
    const interaction::RuntimeDiagnostics& left,
    const interaction::RuntimeDiagnostics& right) {
    return left.state == right.state &&
           left.result == right.result &&
           left.reason == right.reason &&
           left.target == right.target &&
           left.object_state == right.object_state &&
           left.affordance_id == right.affordance_id &&
           left.clip == right.clip &&
           left.frame == right.frame &&
           left.phase == right.phase &&
           left.hand == right.hand &&
           left.total_cost == right.total_cost &&
           left.group_costs == right.group_costs &&
           left.requested_root_correction_m ==
               right.requested_root_correction_m &&
           left.applied_root_correction_m ==
               right.applied_root_correction_m &&
           left.requested_yaw_correction_radians ==
               right.requested_yaw_correction_radians &&
           left.applied_yaw_correction_radians ==
               right.applied_yaw_correction_radians &&
           left.playback_speed == right.playback_speed &&
           left.hand_position_error_m == right.hand_position_error_m &&
           left.hand_orientation_error_radians ==
               right.hand_orientation_error_radians &&
           left.hand_constraint_weight == right.hand_constraint_weight &&
           left.attached == right.attached &&
           left.recorded_carry == right.recorded_carry &&
           left.inactive_arm_targets_locomotion ==
               right.inactive_arm_targets_locomotion &&
           left.inactive_arm_tracks_locomotion ==
               right.inactive_arm_tracks_locomotion &&
           left.pack_available == right.pack_available &&
           exact(left.pickup_source, right.pickup_source) &&
           exact(left.place, right.place);
}

bool exact(
    const interaction::RuntimeOutput& left,
    const interaction::RuntimeOutput& right) {
    return left.owns_pose == right.owns_pose &&
           left.suppress_steering == right.suppress_steering &&
           exact(left.pose, right.pose) &&
           exact(left.object_world, right.object_world) &&
           exact(left.diagnostics, right.diagnostics);
}

bool exact(
    const interaction::MatchCandidate& left,
    const interaction::MatchCandidate& right) {
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

bool exact(
    const interaction::PickEntryRoot& left,
    const interaction::PickEntryRoot& right) {
    return left.world_x == right.world_x &&
           left.world_z == right.world_z &&
           left.world_yaw_radians == right.world_yaw_radians;
}

bool same_bits(
    const interaction::PickEntryRoot& left,
    const interaction::PickEntryRoot& right) {
    return same_float_bits(left.world_x, right.world_x) &&
           same_float_bits(left.world_z, right.world_z) &&
           same_float_bits(
               left.world_yaw_radians, right.world_yaw_radians);
}

bool exact(
    const interaction::PickEntryPreview& left,
    const interaction::PickEntryPreview& right) {
    return left.path_feasible == right.path_feasible &&
           left.match_ready == right.match_ready &&
           left.path_reason == right.path_reason &&
           left.match_reason == right.match_reason &&
           exact(left.prospective_root, right.prospective_root) &&
           left.feasible_entry_frame == right.feasible_entry_frame &&
           left.contact_frame == right.contact_frame &&
           left.total_cost == right.total_cost &&
           exact(left.match_candidate, right.match_candidate) &&
           exact(left.pickup_source, right.pickup_source);
}

void assert_exact(
    const interaction::RuntimeObservation& left,
    const interaction::RuntimeObservation& right) {
    assert(left.state == right.state);
    assert(exact(left.diagnostics, right.diagnostics));
    assert(left.targets.size() == right.targets.size());
    for (size_t index = 0; index < left.targets.size(); ++index) {
        assert(exact(left.targets[index], right.targets[index]));
    }
    assert(left.surfaces.size() == right.surfaces.size());
    for (size_t index = 0; index < left.surfaces.size(); ++index) {
        assert(exact(left.surfaces[index], right.surfaces[index]));
    }
    assert(same_bits(left.caller_snapshot, right.caller_snapshot));
}

interaction::PickEntryRoot live_pick_entry_root(
    const interaction::LocomotionSnapshot& snapshot) {
    constexpr size_t root = static_cast<size_t>(g1_skeleton::Simulation);
    const vec3 forward = quat_mul_vec3(
        snapshot.pose.rotations[root], vec3(0.0F, 0.0F, 1.0F));
    return {
        snapshot.pose.positions[root].x,
        snapshot.pose.positions[root].z,
        std::atan2(forward.x, forward.z),
    };
}

void assert_preview_rejected(
    const interaction::PickEntryPreview& preview,
    interaction::Reason reason) {
    assert(!preview.path_feasible);
    assert(!preview.match_ready);
    assert(preview.path_reason == reason);
    assert(preview.match_reason == reason);
    assert(preview.feasible_entry_frame == -1);
    assert(preview.contact_frame == -1);
    assert(preview.total_cost == 0.0F);
    assert(exact(preview.match_candidate, interaction::MatchCandidate{}));
}

void assert_preview_root(
    const interaction::PickEntryPreview& preview,
    const interaction::PickEntryRoot& root) {
    assert(same_bits(preview.prospective_root, root));
}

void assert_same_next_update_after_preview(
    interaction::InteractionRuntime& previewed_runtime,
    interaction::RuntimeFixture& previewed,
    interaction::InteractionRuntime& untouched_runtime,
    interaction::RuntimeFixture& untouched,
    const interaction::RuntimeInput& input) {
    std::optional<interaction::RuntimeOutput> previewed_output;
    std::optional<interaction::RuntimeOutput> untouched_output;
    bool previewed_threw = false;
    bool untouched_threw = false;
    try {
        previewed_output = previewed_runtime.update(input);
    } catch (const std::out_of_range&) {
        previewed_threw = true;
    }
    try {
        untouched_output = untouched_runtime.update(input);
    } catch (const std::out_of_range&) {
        untouched_threw = true;
    }
    assert(previewed_threw == untouched_threw);
    assert(previewed_output.has_value() == untouched_output.has_value());
    if (previewed_output.has_value()) {
        assert(exact(*previewed_output, *untouched_output));
    }
    assert_exact(
        interaction::observe(
            previewed_runtime, previewed, input.locomotion),
        interaction::observe(
            untouched_runtime, untouched, input.locomotion));
}

interaction::RuntimeInput idle_input(
    const interaction::LocomotionSnapshot& locomotion) {
    interaction::RuntimeInput input{};
    input.dt = kDt;
    input.locomotion = locomotion;
    return input;
}

interaction::RuntimeInput interact_input(
    const interaction::LocomotionSnapshot& locomotion,
    std::optional<interaction::PickRequest> request) {
    interaction::RuntimeInput input = idle_input(locomotion);
    input.interact_pressed = true;
    input.pick_request = request;
    return input;
}

interaction::RuntimeInput place_interact_input(
    const interaction::LocomotionSnapshot& locomotion,
    std::optional<interaction::PlaceRequest> request) {
    interaction::RuntimeInput input = idle_input(locomotion);
    input.interact_pressed = true;
    input.place_request = request;
    return input;
}

interaction::RuntimeInput cancel_input(
    const interaction::LocomotionSnapshot& locomotion) {
    interaction::RuntimeInput input = idle_input(locomotion);
    input.cancel_pressed = true;
    return input;
}

interaction::RuntimeInput reset_input(
    const interaction::LocomotionSnapshot& locomotion) {
    interaction::RuntimeInput input = idle_input(locomotion);
    input.reset_pressed = true;
    return input;
}

interaction::RuntimeOutput advance(
    interaction::InteractionRuntime& runtime,
    const interaction::LocomotionSnapshot& locomotion) {
    return runtime.update(idle_input(locomotion));
}

void assert_valid_hand_constraint_weight(
    const interaction::RuntimeDiagnostics& diagnostics) {
    assert(std::isfinite(diagnostics.hand_constraint_weight));
    assert(diagnostics.hand_constraint_weight >= 0.0F);
    assert(diagnostics.hand_constraint_weight <= 1.0F);
}

interaction::RuntimeOutput advance_until(
    interaction::InteractionRuntime& runtime,
    const interaction::LocomotionSnapshot& locomotion,
    interaction::RuntimeState expected,
    interaction::RuntimeState permitted_before) {
    for (int update = 0; update < kMaximumUpdates; ++update) {
        const interaction::RuntimeOutput output = advance(runtime, locomotion);
        if (output.diagnostics.state == expected) return output;
        assert(output.diagnostics.state == permitted_before);
    }
    assert(false && "interaction runtime did not reach expected state");
    return {};
}

interaction::RuntimeOutput enter_align(
    interaction::InteractionRuntime& runtime,
    const interaction::RuntimeFixture& fixture) {
    using namespace interaction;
    RuntimeOutput output = runtime.update(interact_input(
        fixture.locomotion, fixture.request));
    assert(output.diagnostics.state == RuntimeState::Preflight);
    output = advance(runtime, fixture.locomotion);
    assert(output.diagnostics.state == RuntimeState::Align);
    return output;
}

interaction::RuntimeOutput enter_carry_before_first_update(
    interaction::InteractionRuntime& runtime,
    const interaction::RuntimeFixture& fixture) {
    using namespace interaction;
    RuntimeOutput output = enter_align(runtime, fixture);
    assert(!output.diagnostics.inactive_arm_tracks_locomotion);
    for (int update = 0;
         update < kMaximumUpdates &&
         output.diagnostics.state != RuntimeState::Carry;
         ++update) {
        assert(!output.diagnostics.inactive_arm_tracks_locomotion);
        output = advance(runtime, fixture.locomotion);
    }
    assert(output.diagnostics.state == RuntimeState::Carry);
    assert(!output.diagnostics.inactive_arm_tracks_locomotion);
    return output;
}

void seed_authored_interaction_slots(interaction::RuntimeFixture& fixture) {
    interaction::InteractionTarget* target = fixture.registry.find(
        fixture.request.target);
    assert(target != nullptr);
    target->affordances.front().interaction_slots = {
        {3U, -0.41F, -0.22F, 1.10F},
        {9U, 0.18F, -0.39F, 0.20F},
    };
}

interaction::PlaceRequest place_request_for(
    const interaction::InteractionRuntime& runtime,
    const interaction::RuntimeFixture& fixture,
    uint64_t request_id = 1001U) {
    const interaction::PlaceStagingPreview preview = runtime.preview_place(
        fixture.surface, fixture.place_affordance_id);
    assert(preview.accepted);
    interaction::PlaceRequest request{};
    request.held_target = fixture.request.target;
    request.surface = fixture.surface;
    request.affordance_id = fixture.place_affordance_id;
    request.request_id = request_id;
    request.selection_id = preview.candidate.selection_id;
    return request;
}

interaction::RuntimeOutput enter_place_align(
    interaction::InteractionRuntime& runtime,
    const interaction::RuntimeFixture& fixture,
    interaction::PlaceRequest request) {
    using namespace interaction;
    RuntimeOutput output = runtime.update(place_interact_input(
        fixture.locomotion, request));
    assert(output.diagnostics.state == RuntimeState::PlacePreflight);
    output = advance(runtime, fixture.locomotion);
    assert(output.diagnostics.state == RuntimeState::PlaceAlign);
    assert(output.diagnostics.attached);
    return output;
}

interaction::RuntimeOutput enter_ready_place_align(
    interaction::InteractionRuntime& runtime,
    const interaction::RuntimeFixture& fixture,
    uint64_t request_id = 1001U) {
    return enter_place_align(
        runtime,
        fixture,
        place_request_for(runtime, fixture, request_id));
}

void assert_held_by_original_owner(
    const interaction::RuntimeFixture& fixture) {
    const interaction::InteractionTarget* held =
        fixture.registry.find(fixture.request.target);
    assert(held != nullptr);
    assert(held->state == interaction::ObjectState::Held);
    assert(held->owner_request == fixture.request.request_id);
}

void assert_fresh_carry_next_sample(
    interaction::InteractionRuntime& runtime,
    const interaction::RuntimeFixture& fixture,
    const interaction::RuntimeOutput& recovery,
    const interaction::RuntimeConfig& config = interaction::RuntimeConfig{}) {
    using namespace interaction;
    const InteractionTarget* target = fixture.registry.find(
        fixture.request.target);
    assert(target != nullptr);
    const GraspAffordance* affordance = fixture.registry.find_affordance(
        fixture.request.target, fixture.request.affordance_id);
    assert(affordance != nullptr);
    CarryController expected(
        fixture.database,
        fixture.features,
        classify_carry_ranges(fixture.database, config.carry),
        config.carry,
        config.ik);
    expected.start(
        recovery.pose,
        affordance->hand,
        *affordance,
        recovery.object_world);
    const Pose expected_pose = expected.update(fixture.locomotion, kDt);
    const Transform expected_object = expected.object_world();

    const RuntimeOutput next = advance(runtime, fixture.locomotion);
    assert(next.diagnostics.state == RuntimeState::Carry);
    assert(exact(next.pose, expected_pose));
    assert(exact(next.object_world, expected_object));
    assert(next.diagnostics.attached);
    assert_held_by_original_owner(fixture);
}

void assert_preflight_rejection_preserves_carry(
    interaction::InteractionRuntime& runtime,
    const interaction::RuntimeFixture& fixture,
    const interaction::RuntimeInput& edge,
    interaction::Reason expected_reason) {
    using namespace interaction;
    InteractionRuntime paused_control = runtime;
    const RuntimeDiagnostics before = runtime.diagnostics();
    const InteractionTarget target_before =
        *fixture.registry.find(fixture.request.target);

    RuntimeOutput output = runtime.update(edge);
    assert(output.diagnostics.state == RuntimeState::PlacePreflight);
    const Pose frozen_pose = output.pose;
    const Transform frozen_object = output.object_world;
    output = advance(runtime, fixture.locomotion);
    assert(output.diagnostics.state == RuntimeState::Carry);
    assert(output.diagnostics.result == ResultCode::Rejected);
    assert(output.diagnostics.reason == expected_reason);
    assert(exact(output.pose, frozen_pose));
    assert(exact(output.object_world, frozen_object));
    assert(exact(*fixture.registry.find(fixture.request.target), target_before));
    assert(before.state == RuntimeState::Carry);

    const RuntimeOutput trial_next = advance(runtime, fixture.locomotion);
    const RuntimeOutput control_next = advance(
        paused_control, fixture.locomotion);
    assert(exact(trial_next.pose, control_next.pose));
    assert(exact(trial_next.object_world, control_next.object_world));
    assert(trial_next.diagnostics.state == RuntimeState::Carry);
    assert_held_by_original_owner(fixture);
}

void assert_free(
    const interaction::TargetRegistry& registry,
    interaction::TargetHandle handle) {
    const interaction::InteractionTarget* target = registry.find(handle);
    assert(target != nullptr);
    assert(target->state == interaction::ObjectState::Free);
    assert(target->owner_request == 0U);
}

interaction::RuntimeDiagnostics run_rejected(
    interaction::RuntimeFixture fixture) {
    using namespace interaction;
    InteractionRuntime runtime(
        fixture.database, fixture.features, fixture.registry, RuntimeConfig{});

    RuntimeOutput output = runtime.update(interact_input(
        fixture.locomotion, fixture.request));
    assert(output.diagnostics.state == RuntimeState::Preflight);
    output = advance(runtime, fixture.locomotion);
    assert(output.diagnostics.state == RuntimeState::Locomotion);
    assert(output.diagnostics.result == ResultCode::Rejected);
    assert(!output.diagnostics.attached);
    assert(!output.owns_pose && !output.suppress_steering);
    assert_free(fixture.registry, fixture.request.target);
    return output.diagnostics;
}

interaction::RuntimeFixture post_attach_contact_loss_fixture() {
    using namespace interaction;
    RuntimeFixture fixture = make_runtime_fixture();
    constexpr int32_t kFirstPostContactFrame =
        runtime_fixture_detail::kFramesPerClip +
        runtime_fixture_detail::kContactLocalFrame + 1;
    fixture.database.hand_contacts.at(
        static_cast<size_t>(kFirstPostContactFrame) * 2U + 1U) = 0U;
    return fixture;
}

interaction::RuntimeFixture post_success_contact_loss_fixture(
    int32_t frames_after_hold = 1) {
    using namespace interaction;
    RuntimeFixture fixture = make_runtime_fixture();
    const int32_t loss_frame =
        runtime_fixture_detail::kFramesPerClip +
        runtime_fixture_detail::kHoldLocalFrame + frames_after_hold;
    fixture.database.hand_contacts.at(
        static_cast<size_t>(loss_frame) * 2U + 1U) = 0U;
    return fixture;
}

interaction::RuntimeFixture post_ik_table_crossing_fixture() {
    using namespace interaction;
    RuntimeFixture fixture = make_runtime_fixture();
    InteractionTarget* target = fixture.registry.find(fixture.request.target);
    assert(target != nullptr);
    target->table_size.z = 0.01F;
    GraspAffordance& affordance = target->affordances.at(0);
    affordance.clearance_radius = 0.005F;
    affordance.hand_in_object.position.y -= 0.06F;
    affordance.hand_in_object.position.z += 0.04F;

    const RawQuery raw = build_raw_query(query_input_for(
        fixture, *target, affordance));
    fixture.features.offsets.assign(raw.begin(), raw.end());
    return fixture;
}

interaction::RuntimeFixture entry_blend_table_crossing_fixture() {
    using namespace interaction;
    RuntimeFixture fixture = make_runtime_fixture();
    fixture.locomotion.pose.positions[kRightHandBone] =
        vec3(0.0F, 0.55F, 0.20F);
    const InteractionTarget* target = fixture.registry.find(
        fixture.request.target);
    const GraspAffordance* affordance = fixture.registry.find_affordance(
        fixture.request.target, fixture.request.affordance_id);
    assert(target != nullptr && affordance != nullptr);
    const RawQuery raw = build_raw_query(query_input_for(
        fixture, *target, *affordance));
    fixture.features.offsets.assign(raw.begin(), raw.end());
    return fixture;
}

interaction::RuntimeFixture post_ik_object_crossing_fixture() {
    using namespace interaction;
    RuntimeFixture fixture = make_runtime_fixture();
    InteractionTarget* target = fixture.registry.find(fixture.request.target);
    assert(target != nullptr);
    target->table_world.position.y = 0.0F;
    target->table_size.y = 0.10F;
    target->affordances.at(0).hand_in_object.position.y -= 0.06F;
    target->affordances.at(0).hand_in_object.position.z += 0.04F;
    return fixture;
}

interaction::RuntimeFixture curved_precontact_clearance_fixture() {
    using namespace interaction;
    RuntimeFixture fixture = make_runtime_fixture();
    InteractionTarget* target = fixture.registry.find(fixture.request.target);
    assert(target != nullptr);
    GraspAffordance& affordance = target->affordances.at(0);
    affordance.hand_in_object.position.x += 0.08F;
    affordance.clearance_radius = 0.001F;

    constexpr int32_t kIntermediateFrame =
        runtime_fixture_detail::kFramesPerClip + 18;
    vec3 authored = runtime_fixture_detail::read_bone_position(
        fixture.database, kIntermediateFrame, kRightHandBone);
    authored.z += 0.10F;
    runtime_fixture_detail::write_bone_position(
        fixture.database, kIntermediateFrame, kRightHandBone, authored);
    runtime_fixture_detail::recompute_velocities(fixture.database);

    Pose corrected = pose_at_frame(fixture.database, kIntermediateFrame);
    const WorldPose sampled_world = world_pose(corrected);
    const Transform sampled_hand{
        sampled_world.positions[kRightHandBone],
        sampled_world.rotations[kRightHandBone],
    };
    const WorldPose contact_world = world_pose(pose_at_frame(
        fixture.database,
        runtime_fixture_detail::kFramesPerClip +
            runtime_fixture_detail::kContactLocalFrame));
    const Transform source_contact{
        contact_world.positions[kRightHandBone],
        contact_world.rotations[kRightHandBone],
    };
    const Transform target_hand = compose(
        target->object_world, affordance.hand_in_object);
    constexpr float kReachAlpha = 8.0F / 15.0F;
    Transform requested = sampled_hand;
    requested.position = sampled_hand.position + kReachAlpha *
        (target_hand.position - source_contact.position);
    requested.rotation = quat_nlerp_shortest(
        sampled_hand.rotation,
        quat_normalize(quat_mul(
            quat_mul(
                target_hand.rotation,
                quat_inv(source_contact.rotation)),
            sampled_hand.rotation)),
        kReachAlpha);
    IKConfig ik_config{};
    ik_config.accepted_position_m = 0.01F;
    ik_config.maximum_iterations = 64;
    const IKResult ik = solve_hand_ik(
        corrected, Hand::Right, requested, ik_config);
    assert(ik.accepted);
    const vec3 corrected_hand = world_pose(corrected).positions[
        kRightHandBone];
    target->table_world = {corrected_hand, quat()};
    target->table_size = vec3(0.004F, 0.004F, 0.004F);

    const RawQuery raw = build_raw_query(query_input_for(
        fixture, *target, affordance));
    fixture.features.offsets.assign(raw.begin(), raw.end());
    return fixture;
}

interaction::RuntimeFixture reverse_place_runtime_fixture() {
    using namespace interaction;
    using namespace interaction::runtime_fixture_detail;
    RuntimeFixture fixture = make_runtime_fixture();
    fixture.place_library.recorded.clear();
    for (int32_t local = kContactLocalFrame + 1;
         local <= kHoldLocalFrame;
         ++local) {
        const int32_t frame = kFramesPerClip + local;
        const float alpha = static_cast<float>(
            local - (kContactLocalFrame + 1)) /
            static_cast<float>(
                kHoldLocalFrame - (kContactLocalFrame + 1));
        vec3 object = read_vec3(
            fixture.database.object_positions,
            static_cast<size_t>(frame));
        object.y = 0.80F + 0.15F * alpha;
        write_vec3(
            fixture.database.object_positions,
            static_cast<size_t>(frame),
            object);
        const vec3 root = read_bone_position(
            fixture.database, frame, g1_skeleton::Simulation);
        write_bone_position(
            fixture.database, frame, kRightHandBone, object - root);
        write_bone_position(
            fixture.database, frame, kLeftHandBone, object - root);
    }
    recompute_velocities(fixture.database);
    return fixture;
}

interaction::Pose reachable_runtime_arm_pose(
    interaction::Pose pose,
    float link_length) {
    using namespace interaction;
    for (const HingeJoint& joint : kRightArm) {
        pose.positions[static_cast<size_t>(joint.bone)] = vec3();
        pose.rotations[static_cast<size_t>(joint.bone)] =
            joint.rest_rotation;
    }
    pose.positions[static_cast<size_t>(
        g1_skeleton::RightShoulderPitch)] = vec3(0.0F, 0.0F, 1.0F);
    const quat shoulder_rest = quat_normalize(quat_mul(
        kRightArm[0].rest_rotation,
        kRightArm[1].rest_rotation));
    const vec3 vertical_link = quat_mul_vec3(
        quat_inv(shoulder_rest),
        vec3(0.0F, link_length, 0.0F));
    pose.positions[static_cast<size_t>(g1_skeleton::RightElbow)] =
        vertical_link;
    pose.positions[static_cast<size_t>(g1_skeleton::RightWrist)] =
        vertical_link;
    return pose;
}

interaction::Pose bent_runtime_arm_pose(
    interaction::Pose pose,
    float radians) {
    using namespace interaction;
    pose.rotations[static_cast<size_t>(g1_skeleton::RightShoulderRoll)] =
        quat_normalize(quat_mul(
            kRightArm[1].rest_rotation,
            quat_from_angle_axis(radians, kRightArm[1].axis)));
    pose.rotations[static_cast<size_t>(g1_skeleton::RightWristRoll)] =
        quat_normalize(quat_mul(
            kRightArm[4].rest_rotation,
            quat_from_angle_axis(-2.0F * radians, kRightArm[4].axis)));
    return pose;
}

interaction::Transform runtime_right_hand(
    const interaction::Pose& pose) {
    const interaction::WorldPose world = interaction::world_pose(pose);
    return {
        world.positions[interaction::kRightHandBone],
        world.rotations[interaction::kRightHandBone],
    };
}

void write_runtime_database_pose(
    interaction::Database& database,
    int32_t frame,
    const interaction::Pose& pose) {
    using namespace interaction::runtime_fixture_detail;
    for (size_t bone = 0; bone < g1_skeleton::BoneCount; ++bone) {
        write_bone_position(database, frame, bone, pose.positions[bone]);
        write_bone_rotation(database, frame, bone, pose.rotations[bone]);
    }
}

interaction::RuntimeConfig precomputed_place_runtime_config() {
    interaction::RuntimeConfig config{};
    config.place.timing.entry_blend_seconds = kDt;
    config.ik.accepted_orientation_radians = 0.05F;
    config.ik.maximum_iterations = 64;
    return config;
}

struct PrecomputedRuntimeSentinels {
    uint64_t source_id = 2001U;
    float source_support_height_m = 0.625F;
    float requested_support_height_m = 0.71875F;
    float source_contact_offset_m = 0.129953071F;
    float destination_contact_offset_m = 0.13F;
};

interaction::RuntimeFixture precomputed_place_runtime_fixture(
    PrecomputedRuntimeSentinels sentinels = {}) {
    using namespace interaction;
    using namespace interaction::runtime_fixture_detail;
    RuntimeFixture fixture = reverse_place_runtime_fixture();
    constexpr int32_t clip = 0;
    constexpr int32_t range_start = clip * kFramesPerClip;
    constexpr int32_t contact = range_start + kContactLocalFrame;
    constexpr int32_t lift = range_start + kLiftLocalFrame;
    constexpr int32_t hold = range_start + kHoldLocalFrame;
    constexpr int32_t reverse_start = hold + 4;
    const float source_support = sentinels.source_support_height_m;
    const float requested_support = sentinels.requested_support_height_m;
    const float source_contact_offset = sentinels.source_contact_offset_m;
    const float destination_contact_offset =
        sentinels.destination_contact_offset_m;
    if (source_contact_offset == 0.129953071F) {
        assert(canonical_float_bits(source_contact_offset) == 0x3e05126bU);
    }
    const float authored_hand_correction =
        requested_support + destination_contact_offset -
        (source_support + source_contact_offset);
    constexpr float bend_radians = 0.40F;

    fixture.database.active_hands[static_cast<size_t>(clip)] =
        static_cast<uint8_t>(Hand::Right);
    fixture.database.table_positions[
        static_cast<size_t>(clip) * 3U + 1U] = source_support - 0.35F;
    for (int32_t local = kContactLocalFrame;
         local < kFramesPerClip;
         ++local) {
        const int32_t frame = range_start + local;
        fixture.database.hand_contacts[
            static_cast<size_t>(frame) * 2U + 1U] = 1U;
        if (local > reverse_start) continue;
        const float alpha = local <= kContactLocalFrame + 1
            ? 0.0F : static_cast<float>(
                  local - (kContactLocalFrame + 1)) /
                  static_cast<float>(
                      kHoldLocalFrame - (kContactLocalFrame + 1));
        vec3 object = read_vec3(
            fixture.database.object_positions,
            static_cast<size_t>(frame));
        if (local == kContactLocalFrame) {
            object.y = source_support + source_contact_offset;
        } else if (local >= kHoldLocalFrame) {
            object.y = 0.97F;
        } else {
            object.y = 0.865F + 0.085F * alpha;
        }
        write_vec3(
            fixture.database.object_positions,
            static_cast<size_t>(frame),
            object);
        const vec3 root = read_bone_position(
            fixture.database, frame, g1_skeleton::Simulation);
        write_bone_position(
            fixture.database, frame, kRightHandBone, object - root);
    }

    const Pose contact_pose = pose_at_frame(fixture.database, contact);
    const Transform desired_contact_hand = runtime_right_hand(contact_pose);
    const auto measured_vertical_correction = [&](float link_length) {
        Pose straight = reachable_runtime_arm_pose(
            contact_pose, link_length);
        Pose bent = bent_runtime_arm_pose(straight, bend_radians);
        const vec3 translation =
            desired_contact_hand.position - runtime_right_hand(bent).position;
        straight.positions[static_cast<size_t>(
            g1_skeleton::Simulation)] =
            straight.positions[static_cast<size_t>(
                g1_skeleton::Simulation)] + translation;
        bent.positions[static_cast<size_t>(g1_skeleton::Simulation)] =
            bent.positions[static_cast<size_t>(
                g1_skeleton::Simulation)] + translation;
        return runtime_right_hand(straight).position.y -
            runtime_right_hand(bent).position.y;
    };

    float link_length = 0.75F;
    for (int iteration = 0; iteration < 8; ++iteration) {
        const float measured = measured_vertical_correction(link_length);
        assert(measured > 0.0F);
        link_length *= authored_hand_correction / measured;
    }
    float measured = measured_vertical_correction(link_length);
    for (int adjustment = 0;
         measured != authored_hand_correction && adjustment < 100000;
         ++adjustment) {
        link_length = std::nextafter(
            link_length,
            measured > authored_hand_correction
                ? 0.0F
                : std::numeric_limits<float>::infinity());
        measured = measured_vertical_correction(link_length);
    }
    assert(measured == authored_hand_correction);

    Transform source_contact_hand{};
    Transform target_release_hand{};
    for (int32_t frame = range_start;
         frame < range_start + kFramesPerClip;
         ++frame) {
        const Pose original = pose_at_frame(fixture.database, frame);
        const Transform desired_hand = runtime_right_hand(original);
        Pose straight = reachable_runtime_arm_pose(original, link_length);
        Pose bent = bent_runtime_arm_pose(straight, bend_radians);
        const Transform bent_hand = runtime_right_hand(bent);
        bent.positions[static_cast<size_t>(g1_skeleton::Simulation)] =
            bent.positions[static_cast<size_t>(g1_skeleton::Simulation)] +
            (desired_hand.position - bent_hand.position);
        write_runtime_database_pose(fixture.database, frame, bent);
        const Transform authored_hand = runtime_right_hand(bent);
        if (frame >= contact) {
            write_vec3(
                fixture.database.object_positions,
                static_cast<size_t>(frame),
                authored_hand.position);
            const size_t rotation = static_cast<size_t>(frame) * 4U;
            fixture.database.object_rotations[rotation] =
                authored_hand.rotation.w;
            fixture.database.object_rotations[rotation + 1U] =
                authored_hand.rotation.x;
            fixture.database.object_rotations[rotation + 2U] =
                authored_hand.rotation.y;
            fixture.database.object_rotations[rotation + 3U] =
                authored_hand.rotation.z;
        }
        if (frame == contact) {
            source_contact_hand = authored_hand;
            straight.positions[static_cast<size_t>(
                g1_skeleton::Simulation)] =
                bent.positions[static_cast<size_t>(
                    g1_skeleton::Simulation)];
            target_release_hand = runtime_right_hand(straight);
        }
        set_group_row(
            fixture.features,
            frame,
            {10.0F, 10.0F, 10.0F, 10.0F, 10.0F});
    }
    assert(std::abs(
        source_contact_hand.position.y -
        (source_support + source_contact_offset)) < 0.000001F);
    assert(target_release_hand.position.y -
           source_contact_hand.position.y == authored_hand_correction);
    assert(std::abs(
        target_release_hand.position.x - source_contact_hand.position.x) <
        0.00001F);
    assert(std::abs(
        target_release_hand.position.z - source_contact_hand.position.z) <
        0.00001F);
    recompute_velocities(fixture.database);

    const InteractionTarget* pickup_target = fixture.registry.find(
        fixture.request.target);
    assert(pickup_target != nullptr);

    PlacementSurface destination =
        *fixture.surface_registry.find(fixture.surface);
    destination.surface_world.position.y = requested_support;
    destination.support_volume_world.position.y =
        requested_support - 0.35F;
    destination.surface_world.position.x = target_release_hand.position.x;
    destination.surface_world.position.z = target_release_hand.position.z;
    destination.support_volume_world.position.x =
        target_release_hand.position.x;
    destination.support_volume_world.position.z =
        target_release_hand.position.z;
    destination.affordances.front().object_in_surface = {
        vec3(0.0F, destination_contact_offset, 0.0F),
        target_release_hand.rotation,
    };
    destination.affordances.front().support_point_object = quat_mul_vec3(
        quat_inv(target_release_hand.rotation),
        vec3(0.0F, -destination_contact_offset, 0.0F));
    fixture.surface = fixture.surface_registry.upsert(destination);

    PrecomputedReversedPickupClip row{};
    row.id = sentinels.source_id;
    row.object_profile_id = pickup_target->object_profile_id;
    row.sequence_id = "pickup_table__runtime__001";
    row.clip = clip;
    row.entry_frame = range_start + 10;
    row.contact_frame = contact;
    row.lift_frame = lift;
    row.hold_frame = hold;
    row.reverse_start_frame = reverse_start;
    row.hand = Hand::Right;
    row.source_hand_in_object = Transform{};
    row.source_object_bounds = pickup_target->object_bounds;
    row.source_surface = destination;
    row.source_surface.handle = {502U, 3U};
    row.source_surface.surface_world.position.x =
        source_contact_hand.position.x;
    row.source_surface.surface_world.position.y = source_support;
    row.source_surface.surface_world.position.z =
        source_contact_hand.position.z;
    row.source_surface.support_volume_world.position.x =
        source_contact_hand.position.x;
    row.source_surface.support_volume_world.position.y =
        source_support - 0.35F;
    row.source_surface.support_volume_world.position.z =
        source_contact_hand.position.z;
    row.source_surface.affordances.front().object_in_surface = {
        vec3(0.0F, source_contact_offset, 0.0F),
        source_contact_hand.rotation,
    };
    row.source_surface.affordances.front().support_point_object =
        quat_mul_vec3(
            quat_inv(source_contact_hand.rotation),
            vec3(0.0F, -source_contact_offset, 0.0F));
    row.source_affordance_id = fixture.place_affordance_id;
    row.source_support_height_m = source_support;
    row.source_grasp_height_above_support_m = source_contact_offset;
    fixture.place_library.precomputed_reversed = {row};
    return fixture;
}

void assert_invalid_runtime_config(interaction::RuntimeConfig config) {
    using namespace interaction;
    RuntimeFixture fixture = make_runtime_fixture();
    bool threw = false;
    try {
        InteractionRuntime runtime(
            fixture.database, fixture.features, fixture.registry, config);
        (void)runtime;
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
    assert_free(fixture.registry, fixture.request.target);
}

void assert_both_runtime_constructors_reject_without_mutation(
    interaction::RuntimeConfig config) {
    using namespace interaction;
    for (bool placement_enabled : {false, true}) {
        RuntimeFixture fixture = make_runtime_fixture();
        const InteractionTarget target_before =
            *fixture.registry.find(fixture.request.target);
        const PlacementSurface surface_before =
            *fixture.surface_registry.find(fixture.surface);
        bool threw = false;
        try {
            if (placement_enabled) {
                InteractionRuntime runtime(
                    fixture.database,
                    fixture.features,
                    fixture.registry,
                    fixture.surface_registry,
                    fixture.place_library,
                    config);
                (void)runtime;
            } else {
                InteractionRuntime runtime(
                    fixture.database,
                    fixture.features,
                    fixture.registry,
                    config);
                (void)runtime;
            }
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        assert(threw);
        assert(exact(
            *fixture.registry.find(fixture.request.target), target_before));
        assert(exact(
            *fixture.surface_registry.find(fixture.surface), surface_before));
    }
}

void assert_both_runtime_constructors_accept(
    interaction::RuntimeConfig config) {
    using namespace interaction;
    for (bool placement_enabled : {false, true}) {
        RuntimeFixture fixture = make_runtime_fixture();
        if (placement_enabled) {
            InteractionRuntime runtime(
                fixture.database,
                fixture.features,
                fixture.registry,
                fixture.surface_registry,
                fixture.place_library,
                config);
            (void)runtime;
        } else {
            InteractionRuntime runtime(
                fixture.database,
                fixture.features,
                fixture.registry,
                config);
            (void)runtime;
        }
        assert_free(fixture.registry, fixture.request.target);
    }
}

void test_pick_snapshot_map_is_a_rigid_planar_world_map() {
    using namespace interaction;
    constexpr size_t root = static_cast<size_t>(g1_skeleton::Simulation);
    const LocomotionSnapshot input = make_pick_snapshot_fixture();
    const PickEntryRoot slot{2.25F, -1.75F, 0.70F};

    const runtime_detail::PickSnapshotMap mapped =
        runtime_detail::map_pick_entry_snapshot(input, slot);
    assert(mapped.accepted);
    assert(mapped.reason == Reason::None);

    const vec3 live_forward = quat_mul_vec3(
        input.pose.rotations[root], vec3(0.0F, 0.0F, 1.0F));
    const float live_yaw = std::atan2(live_forward.x, live_forward.z);
    const float yaw_delta = std::atan2(
        std::sin(slot.world_yaw_radians - live_yaw),
        std::cos(slot.world_yaw_radians - live_yaw));
    const quat rotation_delta = quat_from_angle_axis(
        yaw_delta, vec3(0.0F, 1.0F, 0.0F));
    const vec3 slot_position(
        slot.world_x, input.pose.positions[root].y, slot.world_z);
    const vec3 translation_delta =
        slot_position -
        quat_mul_vec3(rotation_delta, input.pose.positions[root]);

    assert(mapped.snapshot.pose.positions[root].x == slot.world_x);
    assert(mapped.snapshot.pose.positions[root].y ==
           input.pose.positions[root].y);
    assert(mapped.snapshot.pose.positions[root].z == slot.world_z);
    assert(near_rotation(
        mapped.snapshot.pose.rotations[root],
        quat_mul(rotation_delta, input.pose.rotations[root])));
    const vec3 mapped_forward = quat_mul_vec3(
        mapped.snapshot.pose.rotations[root],
        vec3(0.0F, 0.0F, 1.0F));
    const float mapped_yaw = std::atan2(
        mapped_forward.x, mapped_forward.z);
    assert(near(
        std::atan2(
            std::sin(mapped_yaw - slot.world_yaw_radians),
            std::cos(mapped_yaw - slot.world_yaw_radians)),
        0.0F));
    assert(near(
        mapped.snapshot.pose.velocities[root],
        quat_mul_vec3(rotation_delta, input.pose.velocities[root])));
    assert(near(
        mapped.snapshot.pose.angular_velocities[root],
        quat_mul_vec3(
            rotation_delta, input.pose.angular_velocities[root])));

    for (size_t index = 0;
         index < input.future_root_positions.size();
         ++index) {
        assert(near(
            mapped.snapshot.future_root_positions[index],
            quat_mul_vec3(
                rotation_delta, input.future_root_positions[index]) +
                translation_delta));
        assert(near_rotation(
            mapped.snapshot.future_root_rotations[index],
            quat_mul(
                rotation_delta, input.future_root_rotations[index])));
    }

    const WorldPose live_world = world_pose(input.pose);
    const WorldPose mapped_world = world_pose(mapped.snapshot.pose);
    for (size_t bone = 0; bone < g1_skeleton::BoneCount; ++bone) {
        assert(near(
            mapped_world.positions[bone],
            quat_mul_vec3(rotation_delta, live_world.positions[bone]) +
                translation_delta,
            5.0e-4F));
        assert(near_rotation(
            mapped_world.rotations[bone],
            quat_mul(rotation_delta, live_world.rotations[bone]),
            5.0e-4F));
        assert(near(
            mapped_world.velocities[bone],
            quat_mul_vec3(rotation_delta, live_world.velocities[bone]),
            8.0e-4F));
        assert(near(
            mapped_world.angular_velocities[bone],
            quat_mul_vec3(
                rotation_delta, live_world.angular_velocities[bone]),
            8.0e-4F));
    }


    for (float scale : {0.9991F, 1.0009F}) {
        LocomotionSnapshot scaled = input;
        scaled.pose.rotations[root] =
            scale * scaled.pose.rotations[root];
        const runtime_detail::PickSnapshotMap scaled_map =
            runtime_detail::map_pick_entry_snapshot(scaled, slot);
        assert(scaled_map.accepted);
        assert(near_rotation(
            scaled_map.snapshot.pose.rotations[root],
            quat_mul(rotation_delta, scaled.pose.rotations[root])));
        assert(near(
            quat_length(scaled_map.snapshot.pose.rotations[root]),
            quat_length(scaled.pose.rotations[root])));
        const float yaw_error = std::atan2(
            std::sin(
                orientation_yaw(
                    scaled_map.snapshot.pose.rotations[root]) -
                slot.world_yaw_radians),
            std::cos(
                orientation_yaw(
                    scaled_map.snapshot.pose.rotations[root]) -
                slot.world_yaw_radians));
        assert(near(yaw_error, 0.0F));
    }
}

void test_pick_snapshot_map_preserves_local_channels_and_input() {
    using namespace interaction;
    constexpr size_t root = static_cast<size_t>(g1_skeleton::Simulation);
    const LocomotionSnapshot input = make_pick_snapshot_fixture();
    const LocomotionSnapshot before = input;
    const runtime_detail::PickSnapshotMap mapped =
        runtime_detail::map_pick_entry_snapshot(
            input, PickEntryRoot{2.25F, -1.75F, 0.70F});
    assert(mapped.accepted);
    assert(exact(input, before));

    for (size_t bone = 0; bone < g1_skeleton::BoneCount; ++bone) {
        if (bone == root) continue;
        assert(exact(
            mapped.snapshot.pose.positions[bone],
            input.pose.positions[bone]));
        assert(exact(
            mapped.snapshot.pose.velocities[bone],
            input.pose.velocities[bone]));
        assert(exact(
            mapped.snapshot.pose.rotations[bone],
            input.pose.rotations[bone]));
        assert(exact(
            mapped.snapshot.pose.angular_velocities[bone],
            input.pose.angular_velocities[bone]));
    }
    assert(mapped.snapshot.pose.hand_dof == input.pose.hand_dof);
    assert(
        mapped.snapshot.pose.hand_dof_velocities ==
        input.pose.hand_dof_velocities);
    assert(mapped.snapshot.pose.foot_contacts == input.pose.foot_contacts);
}

void test_pick_snapshot_map_rejects_every_nonfinite_or_nonunit_input() {
    using namespace interaction;
    constexpr size_t root = static_cast<size_t>(g1_skeleton::Simulation);
    const LocomotionSnapshot baseline = make_pick_snapshot_fixture();
    const PickEntryRoot valid_slot{2.25F, -1.75F, 0.70F};
    const auto require = [](bool condition) {
        if (!condition) {
            throw std::runtime_error("pick snapshot fast-math canary failed");
        }
    };
    const auto assert_rejected = [](
        const LocomotionSnapshot& snapshot, PickEntryRoot slot) {
        const LocomotionSnapshot before = snapshot;
        const runtime_detail::PickSnapshotMap mapped =
            runtime_detail::map_pick_entry_snapshot(snapshot, slot);
        if (mapped.accepted || mapped.reason != Reason::OutOfRange ||
            !exact(mapped.snapshot, LocomotionSnapshot{}) ||
            !same_bits(snapshot, before)) {
            throw std::runtime_error("pick snapshot rejection contract failed");
        }
    };

    const std::array<float, 4> nonfinite{
        float_from_bits(0x7f800000U),
        float_from_bits(0xff800000U),
        float_from_bits(0x7fc00001U),
        float_from_bits(0x7f800001U),
    };
    LocomotionSnapshot field_source = baseline;
    const size_t field_count = snapshot_float_fields(field_source).size();
    for (size_t field = 0; field < field_count; ++field) {
        for (float invalid : nonfinite) {
            LocomotionSnapshot mutated = baseline;
            *snapshot_float_fields(mutated).at(field) = invalid;
            assert_rejected(mutated, valid_slot);
        }
    }

    for (size_t contact = 0; contact < baseline.pose.foot_contacts.size();
         ++contact) {
        LocomotionSnapshot mutated = baseline;
        mutated.pose.foot_contacts[contact] = 2U;
        assert_rejected(mutated, valid_slot);
    }

    for (size_t field = 0; field < 3U; ++field) {
        for (float invalid : nonfinite) {
            PickEntryRoot slot = valid_slot;
            std::array<float*, 3> slot_fields{
                &slot.world_x, &slot.world_z, &slot.world_yaw_radians};
            *slot_fields[field] = invalid;
            assert_rejected(baseline, slot);
        }
    }

    LocomotionSnapshot nonunit = baseline;
    nonunit.pose.rotations[root] = quat(1.01F, 0.0F, 0.0F, 0.0F);
    assert_rejected(nonunit, valid_slot);
    nonunit.pose.rotations[root] = quat(0.0F, 0.0F, 0.0F, 0.0F);
    assert_rejected(nonunit, valid_slot);
    nonunit.pose.rotations[root] = quat(
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max());
    assert_rejected(nonunit, valid_slot);

    LocomotionSnapshot undefined_yaw = baseline;
    constexpr float half_sqrt_two = 0.7071067811865475244F;
    undefined_yaw.pose.rotations[root] =
        quat(half_sqrt_two, half_sqrt_two, 0.0F, 0.0F);
    assert_rejected(undefined_yaw, valid_slot);

    LocomotionSnapshot overflow = baseline;
    overflow.pose.positions[root].x = std::numeric_limits<float>::max();
    assert_rejected(
        overflow,
        PickEntryRoot{
            -std::numeric_limits<float>::max(),
            valid_slot.world_z,
            valid_slot.world_yaw_radians});

    LocomotionSnapshot large_finite_geometry = baseline;
    large_finite_geometry.pose.positions[root + 1U] =
        vec3(1.0e20F, -2.0e20F, 3.0e20F);
    assert_rejected(large_finite_geometry, valid_slot);

    LocomotionSnapshot finite_nonunit_locals = baseline;
    finite_nonunit_locals.pose.rotations[root + 1U] =
        quat(2.0F, 0.0F, 0.0F, 0.0F);
    finite_nonunit_locals.future_root_rotations[0] =
        quat(0.25F, 0.0F, 0.0F, 0.0F);
    require(runtime_detail::map_pick_entry_snapshot(
                finite_nonunit_locals, valid_slot)
                .accepted);
}

void test_pick_snapshot_fingerprint_is_fieldwise_and_canonical() {
    using namespace interaction;
    LocomotionSnapshot input = make_pick_snapshot_fixture();
    const LocomotionSnapshot before = input;
    const uint64_t expected = reference_snapshot_fingerprint(input);
    const uint64_t digest =
        runtime_detail::locomotion_snapshot_fingerprint(input);
    assert(digest == expected);
    assert(digest != 0U);
    assert(exact(input, before));

    const size_t field_count = snapshot_float_fields(input).size();
    for (size_t field = 0; field < field_count; ++field) {
        LocomotionSnapshot mutated = input;
        float& value = *snapshot_float_fields(mutated).at(field);
        value = std::nextafter(
            value, std::numeric_limits<float>::infinity());
        assert(
            runtime_detail::locomotion_snapshot_fingerprint(mutated) !=
            digest);
    }
    for (size_t contact = 0; contact < input.pose.foot_contacts.size();
         ++contact) {
        LocomotionSnapshot mutated = input;
        mutated.pose.foot_contacts[contact] ^= 1U;
        assert(
            runtime_detail::locomotion_snapshot_fingerprint(mutated) !=
            digest);
    }

    LocomotionSnapshot negative_zero = input;
    negative_zero.pose.positions[1].x = -0.0F;
    LocomotionSnapshot positive_zero = negative_zero;
    positive_zero.pose.positions[1].x = 0.0F;
    assert(
        runtime_detail::locomotion_snapshot_fingerprint(negative_zero) ==
        runtime_detail::locomotion_snapshot_fingerprint(positive_zero));

    LocomotionSnapshot opposite_quaternions = input;
    for (quat& rotation : opposite_quaternions.pose.rotations) {
        rotation = -rotation;
    }
    for (quat& rotation : opposite_quaternions.future_root_rotations) {
        rotation = -rotation;
    }
    assert(
        runtime_detail::locomotion_snapshot_fingerprint(
            opposite_quaternions) == digest);

    LocomotionSnapshot reordered = input;
    std::swap(
        reordered.pose.positions[0].x,
        reordered.pose.positions[0].y);
    assert(
        runtime_detail::locomotion_snapshot_fingerprint(reordered) !=
        digest);
}

void test_pick_preview_public_api_reports_path_and_match_separately() {
    using namespace interaction;
    static_assert(std::is_same_v<
        decltype(std::declval<const InteractionRuntime&>().preview_pick(
            std::declval<const LocomotionSnapshot&>(),
            PickEntryRoot{},
            TargetHandle{},
            uint32_t{})),
        PickEntryPreview>);

    RuntimeFixture accepted_fixture = make_runtime_fixture();
    InteractionRuntime accepted_runtime(
        accepted_fixture.database,
        accepted_fixture.features,
        accepted_fixture.registry,
        RuntimeConfig{});
    const PickEntryRoot accepted_root = live_pick_entry_root(
        accepted_fixture.locomotion);
    const PickEntryPreview accepted = accepted_runtime.preview_pick(
        accepted_fixture.locomotion,
        accepted_root,
        accepted_fixture.request.target,
        accepted_fixture.request.affordance_id);
    assert(accepted.path_feasible);
    assert(accepted.match_ready);
    assert(accepted.path_reason == Reason::None);
    assert(accepted.match_reason == Reason::None);
    assert_preview_root(accepted, accepted_root);
    assert(accepted.feasible_entry_frame == 85);
    assert(accepted.contact_frame == 100);
    assert(accepted.match_candidate.entry_frame == 85);
    assert(accepted.match_candidate.contact_frame == 100);
    assert(near(accepted.total_cost, 0.80F / 7.0F));
    assert(accepted.total_cost == accepted.match_candidate.total_cost);

    RuntimeFixture blocked_fixture = make_runtime_fixture();
    blocked_fixture.registry.find(blocked_fixture.request.target)
        ->table_size.z = 1.60F;
    InteractionRuntime blocked_runtime(
        blocked_fixture.database,
        blocked_fixture.features,
        blocked_fixture.registry,
        RuntimeConfig{});
    const PickEntryPreview blocked = blocked_runtime.preview_pick(
        blocked_fixture.locomotion,
        live_pick_entry_root(blocked_fixture.locomotion),
        blocked_fixture.request.target,
        blocked_fixture.request.affordance_id);
    assert_preview_rejected(blocked, Reason::BlockedPath);

    RuntimeFixture costly_fixture = high_cost_fixture();
    InteractionRuntime costly_runtime(
        costly_fixture.database,
        costly_fixture.features,
        costly_fixture.registry,
        RuntimeConfig{});
    const PickEntryPreview costly = costly_runtime.preview_pick(
        costly_fixture.locomotion,
        live_pick_entry_root(costly_fixture.locomotion),
        costly_fixture.request.target,
        costly_fixture.request.affordance_id);
    assert(costly.path_feasible);
    assert(!costly.match_ready);
    assert(costly.path_reason == Reason::None);
    assert(costly.match_reason == Reason::PoorMatch);
    assert(costly.feasible_entry_frame == 85);
    assert(costly.contact_frame == 100);
    assert(near(costly.total_cost, 106.40F, 1.0e-4F));
    assert(exact(costly.match_candidate, MatchCandidate{}));

    RuntimeFixture malformed_fixture = make_runtime_fixture();
    malformed_fixture.features.values.clear();
    InteractionRuntime malformed_runtime(
        malformed_fixture.database,
        malformed_fixture.features,
        malformed_fixture.registry,
        RuntimeConfig{});
    const PickEntryPreview malformed = malformed_runtime.preview_pick(
        malformed_fixture.locomotion,
        live_pick_entry_root(malformed_fixture.locomotion),
        malformed_fixture.request.target,
        malformed_fixture.request.affordance_id);
    assert(malformed.path_feasible);
    assert(!malformed.match_ready);
    assert(malformed.path_reason == Reason::None);
    assert(malformed.match_reason == Reason::OutOfRange);
    assert(malformed.feasible_entry_frame == 85);
    assert(malformed.contact_frame == 100);
    assert(malformed.total_cost == 0.0F);
    assert(exact(malformed.match_candidate, MatchCandidate{}));

    RuntimeFixture fallback_fixture = make_runtime_fixture();
    for (size_t dimension = 0; dimension < kFeatureDimension; ++dimension) {
        fallback_fixture.features.values.at(
            85U * kFeatureDimension + dimension) = 20.0F;
    }
    InteractionRuntime fallback_runtime(
        fallback_fixture.database,
        fallback_fixture.features,
        fallback_fixture.registry,
        RuntimeConfig{});
    const PickEntryPreview fallback = fallback_runtime.preview_pick(
        fallback_fixture.locomotion,
        live_pick_entry_root(fallback_fixture.locomotion),
        fallback_fixture.request.target,
        fallback_fixture.request.affordance_id);
    assert(fallback.path_feasible && fallback.match_ready);
    assert(fallback.feasible_entry_frame == 85);
    assert(fallback.contact_frame == 100);
    assert(fallback.match_candidate.entry_frame == 75);
    assert(fallback.total_cost == fallback.match_candidate.total_cost);
    assert(fallback.total_cost < 9.0F);

    RuntimeFixture mixed_fixture = high_cost_fixture();
    mixed_fixture.database.active_hands.at(0) = 1U;
    mixed_fixture.features.values.resize(
        75U * static_cast<size_t>(kFeatureDimension));
    InteractionRuntime mixed_runtime(
        mixed_fixture.database,
        mixed_fixture.features,
        mixed_fixture.registry,
        RuntimeConfig{});
    const PickEntryPreview mixed = mixed_runtime.preview_pick(
        mixed_fixture.locomotion,
        live_pick_entry_root(mixed_fixture.locomotion),
        mixed_fixture.request.target,
        mixed_fixture.request.affordance_id);
    assert(mixed.path_feasible);
    assert(!mixed.match_ready);
    assert(mixed.path_reason == Reason::None);
    assert(mixed.match_reason == Reason::OutOfRange);
    assert(mixed.feasible_entry_frame == 85);
    assert(mixed.contact_frame == 100);
    assert(mixed.total_cost == 0.0F);
    assert(exact(mixed.match_candidate, MatchCandidate{}));
}

void test_pick_preview_rejects_invalid_runtime_target_and_root_inputs() {
    using namespace interaction;
    RuntimeFixture fixture = make_runtime_fixture();
    const PickEntryRoot root = live_pick_entry_root(fixture.locomotion);

    const InteractionRuntime disabled = InteractionRuntime::disabled(
        Reason::PackUnavailable);
    assert_preview_rejected(
        disabled.preview_pick(
            fixture.locomotion,
            root,
            fixture.request.target,
            fixture.request.affordance_id),
        Reason::PackUnavailable);

    InteractionRuntime non_locomotion(
        fixture.database, fixture.features, fixture.registry, RuntimeConfig{});
    const RuntimeOutput edge = non_locomotion.update(interact_input(
        fixture.locomotion, fixture.request));
    assert(edge.diagnostics.state == RuntimeState::Preflight);
    assert_preview_rejected(
        non_locomotion.preview_pick(
            fixture.locomotion,
            root,
            fixture.request.target,
            fixture.request.affordance_id),
        Reason::TargetUnavailable);

    RuntimeFixture target_fixture = make_runtime_fixture();
    InteractionRuntime target_runtime(
        target_fixture.database,
        target_fixture.features,
        target_fixture.registry,
        RuntimeConfig{});
    assert_preview_rejected(
        target_runtime.preview_pick(
            target_fixture.locomotion,
            live_pick_entry_root(target_fixture.locomotion),
            TargetHandle{},
            target_fixture.request.affordance_id),
        Reason::TargetUnavailable);
    TargetHandle stale = target_fixture.request.target;
    ++stale.generation;
    assert_preview_rejected(
        target_runtime.preview_pick(
            target_fixture.locomotion,
            live_pick_entry_root(target_fixture.locomotion),
            stale,
            target_fixture.request.affordance_id),
        Reason::TargetChanged);
    TargetHandle zero_generation = target_fixture.request.target;
    zero_generation.generation = 0U;
    assert_preview_rejected(
        target_runtime.preview_pick(
            target_fixture.locomotion,
            live_pick_entry_root(target_fixture.locomotion),
            zero_generation,
            target_fixture.request.affordance_id),
        Reason::TargetChanged);
    assert_preview_rejected(
        target_runtime.preview_pick(
            target_fixture.locomotion,
            live_pick_entry_root(target_fixture.locomotion),
            target_fixture.request.target,
            0U),
        Reason::TargetUnavailable);

    const std::array<float, 4> nonfinite{
        float_from_bits(0x7f800000U),
        float_from_bits(0xff800000U),
        float_from_bits(0x7fc00001U),
        float_from_bits(0x7f800001U),
    };
    for (size_t field = 0; field < 3U; ++field) {
        for (float value : nonfinite) {
            PickEntryRoot invalid = live_pick_entry_root(
                target_fixture.locomotion);
            std::array<float*, 3> fields{
                &invalid.world_x,
                &invalid.world_z,
                &invalid.world_yaw_radians,
            };
            *fields[field] = value;
            const PickEntryPreview preview = target_runtime.preview_pick(
                target_fixture.locomotion,
                invalid,
                target_fixture.request.target,
                target_fixture.request.affordance_id);
            assert_preview_rejected(preview, Reason::OutOfRange);
            assert_preview_root(preview, invalid);
        }
    }

    LocomotionSnapshot invalid_snapshot = target_fixture.locomotion;
    invalid_snapshot.pose.rotations[g1_skeleton::Simulation] =
        quat(1.01F, 0.0F, 0.0F, 0.0F);
    assert_preview_rejected(
        target_runtime.preview_pick(
            invalid_snapshot,
            live_pick_entry_root(target_fixture.locomotion),
            target_fixture.request.target,
            target_fixture.request.affordance_id),
        Reason::OutOfRange);
}

void test_pick_preview_is_deterministic_and_const_on_every_outcome() {
    using namespace interaction;
    const auto prove_no_mutation = [](
        RuntimeFixture& previewed_fixture,
        RuntimeFixture& control_fixture,
        bool expect_exception) {
        InteractionRuntime previewed(
            previewed_fixture.database,
            previewed_fixture.features,
            previewed_fixture.registry,
            RuntimeConfig{});
        InteractionRuntime control(
            control_fixture.database,
            control_fixture.features,
            control_fixture.registry,
            RuntimeConfig{});
        const PickEntryRoot root = live_pick_entry_root(
            previewed_fixture.locomotion);
        const RuntimeObservation before = observe(
            previewed, previewed_fixture, previewed_fixture.locomotion);
        std::optional<PickEntryPreview> first;
        std::optional<PickEntryPreview> second;
        for (int attempt = 0; attempt < 2; ++attempt) {
            bool threw = false;
            try {
                const PickEntryPreview preview = previewed.preview_pick(
                    previewed_fixture.locomotion,
                    root,
                    previewed_fixture.request.target,
                    previewed_fixture.request.affordance_id);
                if (attempt == 0) first = preview;
                else second = preview;
            } catch (const std::out_of_range&) {
                threw = true;
            }
            assert(threw == expect_exception);
            assert_exact(
                before,
                observe(
                    previewed,
                    previewed_fixture,
                    previewed_fixture.locomotion));
        }
        if (!expect_exception) {
            assert(first.has_value() && second.has_value());
            assert(exact(*first, *second));
        }
        assert_same_next_update_after_preview(
            previewed,
            previewed_fixture,
            control,
            control_fixture,
            interact_input(
                previewed_fixture.locomotion,
                previewed_fixture.request));
        assert_same_next_update_after_preview(
            previewed,
            previewed_fixture,
            control,
            control_fixture,
            idle_input(previewed_fixture.locomotion));
    };

    RuntimeFixture accepted = make_runtime_fixture();
    RuntimeFixture accepted_control = make_runtime_fixture();
    prove_no_mutation(accepted, accepted_control, false);

    RuntimeFixture rejected = make_runtime_fixture();
    RuntimeFixture rejected_control = make_runtime_fixture();
    rejected.registry.find(rejected.request.target)->table_size.z = 1.60F;
    rejected_control.registry.find(rejected_control.request.target)
        ->table_size.z = 1.60F;
    prove_no_mutation(rejected, rejected_control, false);

    RuntimeFixture exceptional = make_runtime_fixture();
    RuntimeFixture exceptional_control = make_runtime_fixture();
    exceptional.features.offsets.clear();
    exceptional_control.features.offsets.clear();
    prove_no_mutation(exceptional, exceptional_control, true);
}

void test_pick_preview_matches_normal_preflight_for_same_realized_snapshot() {
    using namespace interaction;
    RuntimeFixture fixture = make_runtime_fixture();
    InteractionRuntime runtime(
        fixture.database, fixture.features, fixture.registry, RuntimeConfig{});
    PickEntryRoot root = live_pick_entry_root(fixture.locomotion);
    root.world_x += 0.17F;
    root.world_z += 0.11F;
    root.world_yaw_radians += 0.19F;
    const runtime_detail::PickSnapshotMap realized =
        runtime_detail::map_pick_entry_snapshot(fixture.locomotion, root);
    assert(realized.accepted);
    assert(!exact(realized.snapshot, fixture.locomotion));

    const PickEntryPreview preview = runtime.preview_pick(
        fixture.locomotion,
        root,
        fixture.request.target,
        fixture.request.affordance_id);
    assert(preview.path_feasible && preview.match_ready);
    const PickBuildObservation free_build =
        InteractionRuntimeTestAccess::build_pick_evaluation(
            runtime,
            realized.snapshot,
            fixture.request.target,
            fixture.request.affordance_id,
            true);
    assert(free_build.accepted);
    assert(free_build.reason == Reason::None);

    const InteractionTarget* free_target = fixture.registry.find(
        fixture.request.target);
    assert(free_target != nullptr);
    assert(free_target->state == ObjectState::Free);
    assert(free_target->owner_request == 0U);

    RuntimeOutput output = runtime.update(interact_input(
        realized.snapshot, fixture.request));
    assert(output.diagnostics.state == RuntimeState::Preflight);
    output = runtime.update(idle_input(realized.snapshot));
    assert(output.diagnostics.state == RuntimeState::Align);
    assert(output.diagnostics.result == ResultCode::Accepted);
    assert(output.diagnostics.reason == Reason::None);
    assert(output.diagnostics.clip == preview.match_candidate.clip);
    assert(output.diagnostics.frame == preview.match_candidate.entry_frame);
    assert(output.diagnostics.total_cost == preview.match_candidate.total_cost);
    assert(output.diagnostics.group_costs ==
           preview.match_candidate.group_costs);
    const std::optional<MatchCandidate>& preflight_candidate =
        InteractionRuntimeTestAccess::candidate(runtime);
    assert(preflight_candidate.has_value());
    assert(exact(*preflight_candidate, preview.match_candidate));
    const InteractionTarget* targeted = fixture.registry.find(
        fixture.request.target);
    assert(targeted != nullptr);
    assert(targeted->state == ObjectState::Targeted);
    assert(targeted->owner_request == fixture.request.request_id);
    const PickBuildObservation targeted_build =
        InteractionRuntimeTestAccess::build_pick_evaluation(
            runtime,
            realized.snapshot,
            fixture.request.target,
            fixture.request.affordance_id,
            false);
    assert(targeted_build.accepted);
    assert(targeted_build.reason == Reason::None);
    const PickBuildObservation preview_rule_after_reservation =
        InteractionRuntimeTestAccess::build_pick_evaluation(
            runtime,
            realized.snapshot,
            fixture.request.target,
            fixture.request.affordance_id,
            true);
    assert(!preview_rule_after_reservation.accepted);
    assert(preview_rule_after_reservation.reason == Reason::TargetUnavailable);

    RuntimeFixture blocked_fixture = make_runtime_fixture();
    blocked_fixture.registry.find(blocked_fixture.request.target)
        ->table_size.z = 1.60F;
    InteractionRuntime blocked_runtime(
        blocked_fixture.database,
        blocked_fixture.features,
        blocked_fixture.registry,
        RuntimeConfig{});
    const PickEntryPreview blocked_preview = blocked_runtime.preview_pick(
        blocked_fixture.locomotion,
        live_pick_entry_root(blocked_fixture.locomotion),
        blocked_fixture.request.target,
        blocked_fixture.request.affordance_id);
    assert_preview_rejected(blocked_preview, Reason::BlockedPath);
    RuntimeOutput blocked_output = blocked_runtime.update(interact_input(
        blocked_fixture.locomotion, blocked_fixture.request));
    assert(blocked_output.diagnostics.state == RuntimeState::Preflight);
    blocked_output = blocked_runtime.update(idle_input(
        blocked_fixture.locomotion));
    assert(blocked_output.diagnostics.state == RuntimeState::Locomotion);
    assert(blocked_output.diagnostics.result == ResultCode::Rejected);
    assert(blocked_output.diagnostics.reason == blocked_preview.match_reason);
    const InteractionTarget* released = blocked_fixture.registry.find(
        blocked_fixture.request.target);
    assert(released != nullptr);
    assert(released->state == ObjectState::Free);
    assert(released->owner_request == 0U);
}

void test_pickup_source_provenance_preview_preflight_and_registry_join() {
    using namespace interaction;
    RuntimeFixture fixture = make_runtime_fixture();
    InteractionRuntime uncertified_runtime(
        fixture.database, fixture.features, fixture.registry, RuntimeConfig{});
    const PickEntryRoot root = live_pick_entry_root(fixture.locomotion);
    const PickEntryPreview uncertified = uncertified_runtime.preview_pick(
        fixture.locomotion,
        root,
        fixture.request.target,
        fixture.request.affordance_id);
    assert(uncertified.path_feasible && uncertified.match_ready);
    const PickupSourceProvenance& raw = uncertified.pickup_source.provenance;
    const size_t clip = static_cast<size_t>(
        uncertified.match_candidate.clip);
    assert(raw.clip_ordinal == uncertified.match_candidate.clip);
    assert(raw.range_start == fixture.database.range_starts.at(clip));
    assert(raw.range_stop == fixture.database.range_stops.at(clip));
    assert(raw.entry_global_frame ==
           uncertified.match_candidate.entry_frame);
    assert(raw.contact_global_frame ==
           uncertified.match_candidate.contact_frame);
    assert(raw.lift_global_frame ==
           uncertified.match_candidate.lift_frame);
    assert(raw.hold_global_frame ==
           uncertified.match_candidate.hold_frame);
    assert(raw.active_hand == static_cast<Hand>(
        fixture.database.active_hands.at(clip)));
    assert(raw.object_profile_id ==
           fixture.registry.find(fixture.request.target)->object_profile_id);
    assert(same_bits(raw.object_bounds.center_object, vec3()));
    assert(same_bits(
        raw.object_bounds.half_extents_object,
        vec3(
            0.5F * fixture.database.object_dimensions.at(clip * 3U),
            0.5F * fixture.database.object_dimensions.at(clip * 3U + 1U),
            0.5F * fixture.database.object_dimensions.at(clip * 3U + 2U))));
    assert(same_bits(
        raw.hand_in_object.position,
        vec3(
            fixture.database.grasp_positions_object.at(clip * 3U),
            fixture.database.grasp_positions_object.at(clip * 3U + 1U),
            fixture.database.grasp_positions_object.at(clip * 3U + 2U))));
    assert(same_bits(
        raw.hand_in_object.rotation,
        quat(
            fixture.database.grasp_rotations_object.at(clip * 4U),
            fixture.database.grasp_rotations_object.at(clip * 4U + 1U),
            fixture.database.grasp_rotations_object.at(clip * 4U + 2U),
            fixture.database.grasp_rotations_object.at(clip * 4U + 3U))));
    const Transform source_support = compose(
        Transform{
            vec3(
                fixture.database.table_positions.at(clip * 3U),
                fixture.database.table_positions.at(clip * 3U + 1U),
                fixture.database.table_positions.at(clip * 3U + 2U)),
            quat(
                fixture.database.table_rotations.at(clip * 4U),
                fixture.database.table_rotations.at(clip * 4U + 1U),
                fixture.database.table_rotations.at(clip * 4U + 2U),
                fixture.database.table_rotations.at(clip * 4U + 3U)),
        },
        Transform{
            vec3(
                0.0F,
                0.5F * fixture.database.table_sizes.at(clip * 3U + 1U),
                0.0F),
            quat(),
        });
    assert(same_float_bits(
        raw.source_support_height_m, source_support.position.y));
    assert(uncertified.pickup_source.sequence_id.empty());
    assert(uncertified.pickup_source.object_id.empty());
    assert(uncertified.pickup_source.reverse_start_global_frame == -1);
    assert(uncertified.pickup_source.join_key_sha256.empty());
    assert(uncertified.pickup_source.source_id == 0U);

    CertifiedPickupSourceLocalRow local{};
    local.clip_ordinal = raw.clip_ordinal;
    local.range_start = raw.range_start;
    local.range_stop = raw.range_stop;
    local.entry_local_frame = checked_pickup_global_to_local_frame(
        raw.entry_global_frame, raw.range_start, raw.range_stop);
    local.contact_local_frame = checked_pickup_global_to_local_frame(
        raw.contact_global_frame, raw.range_start, raw.range_stop);
    local.lift_local_frame = checked_pickup_global_to_local_frame(
        raw.lift_global_frame, raw.range_start, raw.range_stop);
    local.hold_local_frame = checked_pickup_global_to_local_frame(
        raw.hold_global_frame, raw.range_start, raw.range_stop);
    local.reverse_start_local_frame = local.hold_local_frame;
    local.active_hand = raw.active_hand;
    local.object_profile_id = raw.object_profile_id;
    local.object_bounds = raw.object_bounds;
    local.hand_in_object = raw.hand_in_object;
    local.source_support_height_m = raw.source_support_height_m;
    local.sequence_id = "pickup_table__synthetic_runtime__001";
    local.object_id = "synthetic_runtime_object";
    local.join_key_sha256 =
        "11223344556677881122334455667788"
        "11223344556677881122334455667788";
    local.source_id = UINT64_C(0x1122334455667788);
    const CertifiedPickupSourceIdentity identity =
        make_certified_pickup_source_identity(local);
    const CertifiedPickupSourceRegistry registry({identity});

    RuntimeFixture certified_fixture = make_runtime_fixture();
    InteractionRuntime certified_runtime(
        certified_fixture.database,
        certified_fixture.features,
        certified_fixture.registry,
        RuntimeConfig{},
        &registry);
    const PickEntryPreview certified = certified_runtime.preview_pick(
        certified_fixture.locomotion,
        live_pick_entry_root(certified_fixture.locomotion),
        certified_fixture.request.target,
        certified_fixture.request.affordance_id);
    assert(certified.match_ready);
    assert(exact(certified.pickup_source, identity));
    assert(exact(
        certified.pickup_source.provenance,
        uncertified.pickup_source.provenance));

    RuntimeOutput output = certified_runtime.update(interact_input(
        certified_fixture.locomotion, certified_fixture.request));
    assert(output.diagnostics.state == RuntimeState::Preflight);
    output = certified_runtime.update(idle_input(
        certified_fixture.locomotion));
    assert(output.diagnostics.state == RuntimeState::Align);
    assert(exact(output.diagnostics.pickup_source, identity));
    assert(exact(
        output.diagnostics.pickup_source.provenance,
        certified.pickup_source.provenance));

    CertifiedPickupSourceIdentity nonmatching = identity;
    nonmatching.provenance.source_support_height_m = std::nextafter(
        nonmatching.provenance.source_support_height_m,
        std::numeric_limits<float>::infinity());
    const CertifiedPickupSourceRegistry nonmatching_registry({nonmatching});
    RuntimeFixture unmatched_fixture = make_runtime_fixture();
    InteractionRuntime unmatched_runtime(
        unmatched_fixture.database,
        unmatched_fixture.features,
        unmatched_fixture.registry,
        RuntimeConfig{},
        &nonmatching_registry);
    const PickEntryPreview unmatched = unmatched_runtime.preview_pick(
        unmatched_fixture.locomotion,
        live_pick_entry_root(unmatched_fixture.locomotion),
        unmatched_fixture.request.target,
        unmatched_fixture.request.affordance_id);
    assert(unmatched.match_ready);
    assert(exact(unmatched.pickup_source.provenance, raw));
    assert(unmatched.pickup_source.sequence_id.empty());
    assert(unmatched.pickup_source.object_id.empty());
    assert(unmatched.pickup_source.reverse_start_global_frame == -1);
    assert(unmatched.pickup_source.join_key_sha256.empty());
    assert(unmatched.pickup_source.source_id == 0U);
}

void test_pick_preview_never_reserves_or_constructs_request_authority() {
    using namespace interaction;
    RuntimeFixture fixture = make_place_runtime_fixture();
    RuntimeFixture control_fixture = make_place_runtime_fixture();
    InteractionRuntime runtime(
        fixture.database,
        fixture.features,
        fixture.registry,
        fixture.surface_registry,
        fixture.place_library,
        RuntimeConfig{});
    InteractionRuntime control(
        control_fixture.database,
        control_fixture.features,
        control_fixture.registry,
        control_fixture.surface_registry,
        control_fixture.place_library,
        RuntimeConfig{});
    const PickRequest request_before = fixture.request;
    const RuntimeObservation before = observe(
        runtime, fixture, fixture.locomotion);
    const PickEntryPreview preview = runtime.preview_pick(
        fixture.locomotion,
        live_pick_entry_root(fixture.locomotion),
        fixture.request.target,
        fixture.request.affordance_id);
    assert(preview.match_ready);
    assert_exact(before, observe(runtime, fixture, fixture.locomotion));
    assert(fixture.request.target == request_before.target);
    assert(fixture.request.affordance_id == request_before.affordance_id);
    assert(fixture.request.request_id == request_before.request_id);
    const InteractionTarget* target = fixture.registry.find(
        fixture.request.target);
    assert(target != nullptr);
    assert(target->state == ObjectState::Free);
    assert(target->owner_request == 0U);
    assert_same_next_update_after_preview(
        runtime,
        fixture,
        control,
        control_fixture,
        interact_input(fixture.locomotion, fixture.request));
    assert_same_next_update_after_preview(
        runtime,
        fixture,
        control,
        control_fixture,
        idle_input(fixture.locomotion));
}

void test_pick_preview_rejection_reason_mapping_is_exact() {
    using namespace interaction;
    const auto check = [](
        InteractionRuntime& runtime,
        RuntimeFixture& fixture,
        const LocomotionSnapshot& snapshot,
        PickEntryRoot root,
        TargetHandle target,
        uint32_t affordance_id,
        Reason reason) {
        const RuntimeObservation before = observe(runtime, fixture, snapshot);
        const PickEntryPreview preview = runtime.preview_pick(
            snapshot, root, target, affordance_id);
        assert_preview_rejected(preview, reason);
        assert_preview_root(preview, root);
        assert_exact(before, observe(runtime, fixture, snapshot));
    };

    RuntimeFixture disabled_fixture = make_runtime_fixture();
    InteractionRuntime disabled = InteractionRuntime::disabled(
        Reason::PackUnavailable);
    check(
        disabled,
        disabled_fixture,
        disabled_fixture.locomotion,
        live_pick_entry_root(disabled_fixture.locomotion),
        disabled_fixture.request.target,
        disabled_fixture.request.affordance_id,
        Reason::PackUnavailable);

    RuntimeFixture state_fixture = make_runtime_fixture();
    InteractionRuntime state_runtime(
        state_fixture.database,
        state_fixture.features,
        state_fixture.registry,
        RuntimeConfig{});
    (void)state_runtime.update(interact_input(
        state_fixture.locomotion, state_fixture.request));
    LocomotionSnapshot invalid_while_busy = state_fixture.locomotion;
    invalid_while_busy.pose.positions[g1_skeleton::Simulation].x =
        float_from_bits(0x7fc00001U);
    check(
        state_runtime,
        state_fixture,
        invalid_while_busy,
        live_pick_entry_root(state_fixture.locomotion),
        state_fixture.request.target,
        state_fixture.request.affordance_id,
        Reason::TargetUnavailable);

    RuntimeFixture lookup_fixture = make_runtime_fixture();
    InteractionRuntime lookup_runtime(
        lookup_fixture.database,
        lookup_fixture.features,
        lookup_fixture.registry,
        RuntimeConfig{});
    const PickEntryRoot lookup_root = live_pick_entry_root(
        lookup_fixture.locomotion);
    check(
        lookup_runtime,
        lookup_fixture,
        lookup_fixture.locomotion,
        lookup_root,
        TargetHandle{},
        lookup_fixture.request.affordance_id,
        Reason::TargetUnavailable);
    check(
        lookup_runtime,
        lookup_fixture,
        lookup_fixture.locomotion,
        lookup_root,
        TargetHandle{999U, 1U},
        lookup_fixture.request.affordance_id,
        Reason::TargetUnavailable);
    TargetHandle stale = lookup_fixture.request.target;
    ++stale.generation;
    check(
        lookup_runtime,
        lookup_fixture,
        lookup_fixture.locomotion,
        lookup_root,
        stale,
        lookup_fixture.request.affordance_id,
        Reason::TargetChanged);
    TargetHandle zero_generation = lookup_fixture.request.target;
    zero_generation.generation = 0U;
    check(
        lookup_runtime,
        lookup_fixture,
        lookup_fixture.locomotion,
        lookup_root,
        zero_generation,
        lookup_fixture.request.affordance_id,
        Reason::TargetChanged);
    check(
        lookup_runtime,
        lookup_fixture,
        lookup_fixture.locomotion,
        lookup_root,
        lookup_fixture.request.target,
        0U,
        Reason::TargetUnavailable);
    check(
        lookup_runtime,
        lookup_fixture,
        lookup_fixture.locomotion,
        lookup_root,
        lookup_fixture.request.target,
        999U,
        Reason::TargetUnavailable);

    for (ObjectState state : {
             ObjectState::Targeted,
             ObjectState::Attached,
             ObjectState::Held}) {
        RuntimeFixture nonfree_fixture = make_runtime_fixture();
        InteractionTarget* target = nonfree_fixture.registry.find(
            nonfree_fixture.request.target);
        assert(target != nullptr);
        target->state = state;
        target->owner_request = 1234U;
        InteractionRuntime nonfree_runtime(
            nonfree_fixture.database,
            nonfree_fixture.features,
            nonfree_fixture.registry,
            RuntimeConfig{});
        check(
            nonfree_runtime,
            nonfree_fixture,
            nonfree_fixture.locomotion,
            live_pick_entry_root(nonfree_fixture.locomotion),
            nonfree_fixture.request.target,
            nonfree_fixture.request.affordance_id,
            Reason::TargetUnavailable);
    }
    RuntimeFixture owned_free_fixture = make_runtime_fixture();
    InteractionTarget* owned_free_target = owned_free_fixture.registry.find(
        owned_free_fixture.request.target);
    assert(owned_free_target != nullptr);
    owned_free_target->state = ObjectState::Free;
    owned_free_target->owner_request = 1234U;
    InteractionRuntime owned_free_runtime(
        owned_free_fixture.database,
        owned_free_fixture.features,
        owned_free_fixture.registry,
        RuntimeConfig{});
    check(
        owned_free_runtime,
        owned_free_fixture,
        owned_free_fixture.locomotion,
        live_pick_entry_root(owned_free_fixture.locomotion),
        owned_free_fixture.request.target,
        owned_free_fixture.request.affordance_id,
        Reason::TargetUnavailable);

    RuntimeFixture map_fixture = make_runtime_fixture();
    InteractionRuntime map_runtime(
        map_fixture.database,
        map_fixture.features,
        map_fixture.registry,
        RuntimeConfig{});
    for (size_t field = 0; field < 3U; ++field) {
        PickEntryRoot invalid = live_pick_entry_root(map_fixture.locomotion);
        std::array<float*, 3> fields{
            &invalid.world_x,
            &invalid.world_z,
            &invalid.world_yaw_radians,
        };
        *fields[field] = float_from_bits(0x7f800000U);
        check(
            map_runtime,
            map_fixture,
            map_fixture.locomotion,
            invalid,
            map_fixture.request.target,
            map_fixture.request.affordance_id,
            Reason::OutOfRange);
    }
    LocomotionSnapshot nonfinite_snapshot = map_fixture.locomotion;
    nonfinite_snapshot.pose.velocities[1].y = float_from_bits(0x7fc00001U);
    check(
        map_runtime,
        map_fixture,
        nonfinite_snapshot,
        live_pick_entry_root(map_fixture.locomotion),
        map_fixture.request.target,
        map_fixture.request.affordance_id,
        Reason::OutOfRange);
    LocomotionSnapshot nonunit = map_fixture.locomotion;
    nonunit.pose.rotations[g1_skeleton::Simulation] =
        quat(1.01F, 0.0F, 0.0F, 0.0F);
    check(
        map_runtime,
        map_fixture,
        nonunit,
        live_pick_entry_root(map_fixture.locomotion),
        map_fixture.request.target,
        map_fixture.request.affordance_id,
        Reason::OutOfRange);
    LocomotionSnapshot undefined_yaw = map_fixture.locomotion;
    constexpr float half_sqrt_two = 0.7071067811865475244F;
    undefined_yaw.pose.rotations[g1_skeleton::Simulation] =
        quat(half_sqrt_two, half_sqrt_two, 0.0F, 0.0F);
    check(
        map_runtime,
        map_fixture,
        undefined_yaw,
        live_pick_entry_root(map_fixture.locomotion),
        map_fixture.request.target,
        map_fixture.request.affordance_id,
        Reason::OutOfRange);
    LocomotionSnapshot overflow = map_fixture.locomotion;
    overflow.pose.positions[g1_skeleton::Simulation].x =
        std::numeric_limits<float>::max();
    PickEntryRoot overflow_root = live_pick_entry_root(map_fixture.locomotion);
    overflow_root.world_x = -std::numeric_limits<float>::max();
    check(
        map_runtime,
        map_fixture,
        overflow,
        overflow_root,
        map_fixture.request.target,
        map_fixture.request.affordance_id,
        Reason::OutOfRange);
}

void test_frozen_public_contract_and_defaults() {
    using namespace interaction;

    static_assert(std::is_same_v<
        std::underlying_type_t<RuntimeState>, uint8_t>);
    static_assert(static_cast<uint8_t>(RuntimeState::Disabled) == 0U);
    static_assert(static_cast<uint8_t>(RuntimeState::Locomotion) == 1U);
    static_assert(static_cast<uint8_t>(RuntimeState::Preflight) == 2U);
    static_assert(static_cast<uint8_t>(RuntimeState::Align) == 3U);
    static_assert(static_cast<uint8_t>(RuntimeState::PickupReplay) == 4U);
    static_assert(static_cast<uint8_t>(RuntimeState::Hold) == 5U);
    static_assert(static_cast<uint8_t>(RuntimeState::Carry) == 6U);
    static_assert(static_cast<uint8_t>(RuntimeState::PlacePreflight) == 7U);
    static_assert(static_cast<uint8_t>(RuntimeState::PlaceAlign) == 8U);
    static_assert(static_cast<uint8_t>(RuntimeState::PlaceReplay) == 9U);
    static_assert(static_cast<uint8_t>(RuntimeState::PlaceRelease) == 10U);

    static_assert(std::is_same_v<decltype(RuntimeConfig{}.matcher), MatchConfig>);
    static_assert(std::is_same_v<
        decltype(RuntimeConfig{}.playback), PlaybackConfig>);
    static_assert(std::is_same_v<decltype(RuntimeConfig{}.ik), IKConfig>);
    static_assert(std::is_same_v<
        decltype(RuntimeConfig{}.attachment), AttachmentConfig>);
    static_assert(std::is_same_v<decltype(RuntimeConfig{}.carry), CarryConfig>);
    static_assert(std::is_same_v<
        decltype(RuntimeConfig{}.place), PlaceControllerConfig>);
    static_assert(std::is_same_v<decltype(RuntimeInput{}.dt), float>);
    static_assert(std::is_same_v<
        decltype(RuntimeInput{}.locomotion), LocomotionSnapshot>);
    static_assert(std::is_same_v<
        decltype(RuntimeInput{}.pick_request), std::optional<PickRequest>>);
    static_assert(std::is_same_v<
        decltype(RuntimeInput{}.place_request), std::optional<PlaceRequest>>);
    static_assert(std::is_same_v<
        decltype(RuntimeDiagnostics{}.state), RuntimeState>);
    static_assert(std::is_same_v<
        decltype(RuntimeDiagnostics{}.result), ResultCode>);
    static_assert(std::is_same_v<decltype(RuntimeDiagnostics{}.reason), Reason>);
    static_assert(std::is_same_v<
        decltype(RuntimeDiagnostics{}.target), TargetHandle>);
    static_assert(std::is_same_v<
        decltype(RuntimeDiagnostics{}.object_state), ObjectState>);
    static_assert(std::is_same_v<
        decltype(RuntimeDiagnostics{}.pickup_source),
        CertifiedPickupSourceIdentity>);
    static_assert(std::is_same_v<
        decltype(PickEntryPreview{}.pickup_source),
        CertifiedPickupSourceIdentity>);
    static_assert(std::is_same_v<decltype(RuntimeOutput{}.pose), Pose>);
    static_assert(std::is_same_v<
        decltype(RuntimeOutput{}.object_world), Transform>);
    static_assert(std::is_same_v<
        decltype(RuntimeOutput{}.diagnostics), RuntimeDiagnostics>);
    static_assert(std::is_constructible_v<
        InteractionRuntime,
        const Database&,
        const Features&,
        TargetRegistry&,
        RuntimeConfig>);
    static_assert(std::is_constructible_v<
        InteractionRuntime,
        const Database&,
        const Features&,
        TargetRegistry&,
        RuntimeConfig,
        const CertifiedPickupSourceRegistry*>);
    static_assert(std::is_constructible_v<
        InteractionRuntime,
        const Database&,
        const Features&,
        TargetRegistry&,
        PlacementSurfaceRegistry&,
        const PlaceMotionLibrary&,
        RuntimeConfig>);
    static_assert(std::is_constructible_v<
        InteractionRuntime,
        const Database&,
        const Features&,
        TargetRegistry&,
        PlacementSurfaceRegistry&,
        const PlaceMotionLibrary&,
        RuntimeConfig,
        const CertifiedPickupSourceRegistry*>);
    static_assert(std::is_same_v<
        decltype(std::declval<const InteractionRuntime&>().preview_place(
            SurfaceHandle{}, uint32_t{})),
        PlaceStagingPreview>);
    static_assert(std::is_same_v<
        decltype(&InteractionRuntime::disabled),
        InteractionRuntime (*)(Reason)>);
    static_assert(std::is_same_v<
        decltype(&InteractionRuntime::state),
        RuntimeState (InteractionRuntime::*)() const>);
    static_assert(std::is_same_v<
        decltype(&InteractionRuntime::diagnostics),
        const RuntimeDiagnostics& (InteractionRuntime::*)() const>);
    static_assert(std::is_same_v<
        decltype(RuntimeDiagnostics{}.hand_constraint_weight), float>);
    static_assert(std::is_same_v<
        decltype(RuntimeDiagnostics{}.inactive_arm_targets_locomotion),
        bool>);
    static_assert(std::is_same_v<
        decltype(RuntimeDiagnostics{}.inactive_arm_tracks_locomotion), bool>);
    static_assert(std::is_same_v<
        decltype(&InteractionRuntime::update),
        RuntimeOutput (InteractionRuntime::*)(const RuntimeInput&)>);

    const PlaybackConfig playback{};
    assert(playback.canonical_fps == 25.0F);
    assert(playback.speed == 1.0F);
    assert(playback.minimum_speed == 0.85F);
    assert(playback.maximum_speed == 1.15F);
    assert(playback.entry_blend_seconds == 0.25F);
    assert(playback.maximum_alignment_seconds == 1.00F);
    assert(playback.commit_horizon_seconds == 0.50F);

    const RuntimeInput input{};
    assert(input.dt == 0.0F);
    assert(!input.interact_pressed && !input.pick_request.has_value());
    assert(!input.cancel_pressed && !input.reset_pressed);

    const RuntimeDiagnostics diagnostics{};
    assert(diagnostics.state == RuntimeState::Disabled);
    assert(diagnostics.result == ResultCode::None);
    assert(diagnostics.reason == Reason::None);
    assert(diagnostics.object_state == ObjectState::Free);
    assert(diagnostics.clip == -1 && diagnostics.frame == -1);
    assert(diagnostics.hand_constraint_weight == 0.0F);
    assert(!diagnostics.attached && !diagnostics.recorded_carry);
    assert(!diagnostics.inactive_arm_targets_locomotion);
    assert(!diagnostics.inactive_arm_tracks_locomotion);
    assert(!diagnostics.pack_available);
    assert(diagnostics.place.source_id == 0U);
    assert(diagnostics.place.source_support_height_m == 0.0F);
    assert(diagnostics.place.requested_support_height_m == 0.0F);
    assert(diagnostics.place.target_support_height_m == 0.0F);
    assert(diagnostics.place.requested_vertical_correction_m == 0.0F);
    assert(diagnostics.place.applied_vertical_correction_m == 0.0F);

    const RuntimeOutput output{};
    assert(!output.owns_pose && !output.suppress_steering);
}

void assert_place_immutable_provenance(
    const interaction::RuntimePlaceDiagnostics& diagnostics,
    uint64_t source_id,
    float requested_support_height_m,
    float source_support_height_m,
    float target_support_height_m,
    float requested_vertical_correction_m) {
    assert(diagnostics.source_id == source_id);
    assert(diagnostics.requested_support_height_m ==
           requested_support_height_m);
    assert(diagnostics.source_support_height_m == source_support_height_m);
    assert(diagnostics.target_support_height_m == target_support_height_m);
    assert(diagnostics.requested_vertical_correction_m ==
           requested_vertical_correction_m);
}

struct PrecomputedRuntimeWitness {
    interaction::PlaceCandidate candidate{};
    interaction::RuntimePlaceDiagnostics preflight{};
    interaction::RuntimePlaceDiagnostics accepted_preflight{};
    interaction::RuntimePlaceDiagnostics full_replay{};
    float measured_full_applied_vertical_correction_m = 0.0F;
    bool align_applied_vertical_correction_zero = true;
    bool first_replay_applied_vertical_correction_zero = false;
};

PrecomputedRuntimeWitness run_precomputed_runtime_witness(
    PrecomputedRuntimeSentinels sentinels) {
    using namespace interaction;
    RuntimeFixture fixture = precomputed_place_runtime_fixture(sentinels);
    InteractionRuntime runtime(
        fixture.database,
        fixture.features,
        fixture.registry,
        fixture.surface_registry,
        fixture.place_library,
        precomputed_place_runtime_config());
    enter_carry_before_first_update(runtime, fixture);
    const std::optional<MatchCandidate>& pickup_candidate =
        InteractionRuntimeTestAccess::candidate(runtime);
    assert(pickup_candidate.has_value());
    assert(pickup_candidate->clip == 1);

    const std::optional<PlaceMatchInput> direct_input =
        InteractionRuntimeTestAccess::place_match_input(
            runtime, fixture.surface, fixture.place_affordance_id);
    assert(direct_input.has_value());
    const PlaceStagingPreview free_preview = preview_place_motion(
        *direct_input);
    const PlaceStagingPreview runtime_preview = runtime.preview_place(
        fixture.surface, fixture.place_affordance_id);
    assert(free_preview.accepted && free_preview.ready);
    assert(runtime_preview.accepted && runtime_preview.ready);
    assert(exact(free_preview.candidate, runtime_preview.candidate));
    assert(free_preview.candidate.selection_id ==
           runtime_preview.candidate.selection_id);
    assert(free_preview.candidate.mode ==
           PlaceMotionMode::PrecomputedReversedPickup);
    assert(free_preview.candidate.clip == 0);
    assert(free_preview.candidate.source_id == sentinels.source_id);
    assert(free_preview.candidate.requested_support_height_m ==
           sentinels.requested_support_height_m);
    assert(free_preview.candidate.source_support_height_m ==
           sentinels.source_support_height_m);
    assert(free_preview.candidate.target_support_height_m ==
           sentinels.requested_support_height_m);
    const float requested_vertical_correction_m =
        sentinels.requested_support_height_m -
        sentinels.source_support_height_m;
    assert(free_preview.candidate.requested_vertical_correction_m ==
           requested_vertical_correction_m);

    const PlaceRequest request = place_request_for(runtime, fixture, 2002U);
    RuntimeOutput output = runtime.update(place_interact_input(
        fixture.locomotion, request));
    assert(output.diagnostics.state == RuntimeState::PlacePreflight);
    assert(exact(
        free_preview.candidate,
        output.diagnostics.place.preview.candidate));
    assert_place_immutable_provenance(
        output.diagnostics.place,
        sentinels.source_id,
        sentinels.requested_support_height_m,
        sentinels.source_support_height_m,
        sentinels.requested_support_height_m,
        requested_vertical_correction_m);
    assert(output.diagnostics.place.applied_vertical_correction_m == 0.0F);
    const RuntimePlaceDiagnostics preflight = output.diagnostics.place;

    output = advance(runtime, fixture.locomotion);
    assert(output.diagnostics.state == RuntimeState::PlaceAlign);
    assert(exact(
        free_preview.candidate,
        output.diagnostics.place.preview.candidate));
    assert_place_immutable_provenance(
        output.diagnostics.place,
        sentinels.source_id,
        sentinels.requested_support_height_m,
        sentinels.source_support_height_m,
        sentinels.requested_support_height_m,
        requested_vertical_correction_m);
    assert(output.diagnostics.place.applied_vertical_correction_m == 0.0F);
    const RuntimePlaceDiagnostics accepted_preflight =
        output.diagnostics.place;

    bool saw_replay = false;
    bool saw_first_replay = false;
    bool saw_nonzero_replay = false;
    bool saw_full_replay = false;
    bool align_applied_zero = true;
    bool first_replay_applied_zero = false;
    RuntimeOutput full_output{};
    float previous_replay_applied = 0.0F;
    for (int update = 0; update < kMaximumUpdates; ++update) {
        const RuntimeState state = output.diagnostics.state;
        if (state == RuntimeState::PlaceAlign ||
            state == RuntimeState::PlaceReplay ||
            state == RuntimeState::PlaceRelease) {
            assert_place_immutable_provenance(
                output.diagnostics.place,
                sentinels.source_id,
                sentinels.requested_support_height_m,
                sentinels.source_support_height_m,
                sentinels.requested_support_height_m,
                requested_vertical_correction_m);
        }
        if (state == RuntimeState::PlaceAlign) {
            align_applied_zero = align_applied_zero &&
                output.diagnostics.place.applied_vertical_correction_m ==
                    0.0F;
        }
        if (state == RuntimeState::PlaceReplay) {
            saw_replay = true;
            const float applied = output.diagnostics.place
                .applied_vertical_correction_m;
            if (!saw_first_replay) {
                first_replay_applied_zero = applied == 0.0F;
                saw_first_replay = true;
            } else {
                assert(applied >= previous_replay_applied);
            }
            if (applied != 0.0F && !saw_nonzero_replay) {
                const double source_frame_exact =
                    output.diagnostics.place.source_frame_exact;
                assert(source_frame_exact == std::floor(source_frame_exact));
                const Transform uncorrected_hand = compose(
                    free_preview.candidate.scene_from_source,
                    runtime_right_hand(pose_at_frame(
                        fixture.database,
                        static_cast<int32_t>(source_frame_exact))));
                const float measured_applied =
                    runtime_right_hand(output.pose).position.y -
                    uncorrected_hand.position.y;
                assert(applied == measured_applied);
                saw_nonzero_replay = true;
            }
            previous_replay_applied = applied;
        }
        if (state == RuntimeState::PlaceRelease) {
            full_output = output;
            saw_full_replay = true;
            break;
        }
        assert(state != RuntimeState::Locomotion);
        output = advance(runtime, fixture.locomotion);
    }
    assert(saw_replay);
    assert(saw_first_replay);
    assert(saw_nonzero_replay);
    assert(saw_full_replay);
    assert(full_output.diagnostics.place.released);
    assert(full_output.diagnostics.place.source_frame ==
           free_preview.candidate.release_frame);

    const Transform uncorrected_release_hand = compose(
        free_preview.candidate.scene_from_source,
        runtime_right_hand(pose_at_frame(
            fixture.database, free_preview.candidate.release_frame)));
    const float measured_full_applied =
        runtime_right_hand(full_output.pose).position.y -
        uncorrected_release_hand.position.y;
    assert(full_output.diagnostics.place.applied_vertical_correction_m ==
           measured_full_applied);

    const RuntimeOutput after_ack = advance(runtime, fixture.locomotion);
    assert(after_ack.diagnostics.state == RuntimeState::PlaceRelease);
    assert(after_ack.diagnostics.place.applied_vertical_correction_m ==
           full_output.diagnostics.place.applied_vertical_correction_m);

    return {
        free_preview.candidate,
        preflight,
        accepted_preflight,
        full_output.diagnostics.place,
        measured_full_applied,
        align_applied_zero,
        first_replay_applied_zero,
    };
}

void test_runtime_preview_exposes_precomputed_provenance_boundary() {
    const PrecomputedRuntimeWitness baseline =
        run_precomputed_runtime_witness({});
    assert(baseline.candidate.source_id == 2001U);
    assert(baseline.candidate.clip == 0);
    assert(baseline.full_replay.applied_vertical_correction_m == 0.09375F);
    assert(baseline.measured_full_applied_vertical_correction_m ==
           0.09375F);
}

void test_place_preflight_and_replay_preserve_precomputed_provenance() {
    const PrecomputedRuntimeWitness baseline =
        run_precomputed_runtime_witness({});

    PrecomputedRuntimeSentinels identity_values{};
    identity_values.source_id = 2111U;
    const PrecomputedRuntimeWitness identity =
        run_precomputed_runtime_witness(identity_values);
    assert(identity.candidate.source_id == 2111U);
    assert(identity.candidate.source_support_height_m ==
           baseline.candidate.source_support_height_m);
    assert(identity.candidate.requested_support_height_m ==
           baseline.candidate.requested_support_height_m);
    assert(identity.candidate.target_support_height_m ==
           baseline.candidate.target_support_height_m);
    assert(identity.candidate.requested_vertical_correction_m ==
           baseline.candidate.requested_vertical_correction_m);
    assert(identity.full_replay.applied_vertical_correction_m ==
           baseline.full_replay.applied_vertical_correction_m);

    PrecomputedRuntimeSentinels shifted_values{};
    shifted_values.source_support_height_m -= 0.03125F;
    shifted_values.requested_support_height_m -= 0.03125F;
    shifted_values.source_contact_offset_m += 0.03125F;
    shifted_values.destination_contact_offset_m += 0.03125F;
    const PrecomputedRuntimeWitness shifted =
        run_precomputed_runtime_witness(shifted_values);
    assert(shifted.candidate.source_support_height_m == 0.59375F);
    assert(shifted.candidate.requested_support_height_m == 0.6875F);
    assert(shifted.candidate.target_support_height_m == 0.6875F);
    assert(shifted.candidate.requested_vertical_correction_m ==
           baseline.candidate.requested_vertical_correction_m);
    assert(shifted.full_replay.applied_vertical_correction_m ==
           baseline.full_replay.applied_vertical_correction_m);

    PrecomputedRuntimeSentinels provenance_values{};
    provenance_values.source_support_height_m -= 0.0078125F;
    provenance_values.source_contact_offset_m += 0.0078125F;
    const PrecomputedRuntimeWitness provenance =
        run_precomputed_runtime_witness(provenance_values);
    assert(provenance.candidate.source_support_height_m == 0.6171875F);
    assert(provenance.candidate.requested_support_height_m ==
           baseline.candidate.requested_support_height_m);
    assert(provenance.candidate.target_support_height_m ==
           baseline.candidate.target_support_height_m);
    assert(provenance.candidate.requested_vertical_correction_m ==
           0.1015625F);
    assert(provenance.full_replay.applied_vertical_correction_m ==
           baseline.full_replay.applied_vertical_correction_m);
    assert(provenance.full_replay.requested_vertical_correction_m !=
           provenance.full_replay.applied_vertical_correction_m);

    PrecomputedRuntimeSentinels applied_values{};
    applied_values.destination_contact_offset_m -= 0.0078125F;
    const PrecomputedRuntimeWitness applied =
        run_precomputed_runtime_witness(applied_values);
    assert(applied.candidate.source_support_height_m ==
           baseline.candidate.source_support_height_m);
    assert(applied.candidate.requested_support_height_m ==
           baseline.candidate.requested_support_height_m);
    assert(applied.candidate.target_support_height_m ==
           baseline.candidate.target_support_height_m);
    assert(applied.candidate.requested_vertical_correction_m ==
           baseline.candidate.requested_vertical_correction_m);
    assert(applied.full_replay.applied_vertical_correction_m !=
           baseline.full_replay.applied_vertical_correction_m);
    assert(applied.full_replay.applied_vertical_correction_m ==
           applied.measured_full_applied_vertical_correction_m);
    assert(baseline.align_applied_vertical_correction_zero);
    assert(baseline.first_replay_applied_vertical_correction_zero);
}

void test_place_preview_and_collapsed_success_lifecycle() {
    using namespace interaction;
    RuntimeFixture fixture = make_runtime_fixture();
    InteractionRuntime runtime(
        fixture.database,
        fixture.features,
        fixture.registry,
        fixture.surface_registry,
        fixture.place_library,
        RuntimeConfig{});

    const RuntimeDiagnostics before_preview = runtime.diagnostics();
    const InteractionTarget before_target =
        *fixture.registry.find(fixture.request.target);
    const PlacementSurface before_surface =
        *fixture.surface_registry.find(fixture.surface);
    const PlaceStagingPreview before_carry = runtime.preview_place(
        fixture.surface, fixture.place_affordance_id);
    assert(!before_carry.accepted);
    assert(before_carry.reason == Reason::OutOfRange);
    assert(exact(runtime.diagnostics(), before_preview));
    assert(exact(
        *fixture.registry.find(fixture.request.target), before_target));
    assert(exact(
        *fixture.surface_registry.find(fixture.surface), before_surface));

    RuntimeOutput output = enter_carry_before_first_update(runtime, fixture);
    const RuntimeOutput frozen = output;
    const RuntimeDiagnostics carry_diagnostics = runtime.diagnostics();
    const InteractionTarget carry_target_before =
        *fixture.registry.find(fixture.request.target);
    const PlacementSurface carry_surface_before =
        *fixture.surface_registry.find(fixture.surface);
    const PlaceStagingPreview first = runtime.preview_place(
        fixture.surface, fixture.place_affordance_id);
    const PlaceStagingPreview second = runtime.preview_place(
        fixture.surface, fixture.place_affordance_id);
    assert(first.accepted);
    assert(first.ready);
    assert(first.candidate.selection_id != 0U);
    assert(first.ik_config_fingerprint != 0U);
    assert(exact(first, second));
    assert(exact(runtime.diagnostics(), carry_diagnostics));
    assert(exact(
        *fixture.registry.find(fixture.request.target), carry_target_before));
    assert(exact(
        *fixture.surface_registry.find(fixture.surface), carry_surface_before));

    PlaceRequest request{};
    request.held_target = fixture.request.target;
    request.surface = fixture.surface;
    request.affordance_id = fixture.place_affordance_id;
    request.request_id = 1001U;
    request.selection_id = first.candidate.selection_id;
    output = runtime.update(place_interact_input(fixture.locomotion, request));
    assert(output.diagnostics.state == RuntimeState::PlacePreflight);
    assert(output.owns_pose && output.suppress_steering);
    assert(output.diagnostics.attached);
    assert(exact(output.pose, frozen.pose));
    assert(exact(output.object_world, frozen.object_world));
    assert(exact(output.diagnostics.place.preview, first));
    assert(output.diagnostics.place.selection_id ==
           first.candidate.selection_id);

    int place_preflight_publications = 1;
    output = advance(runtime, fixture.locomotion);
    assert(output.diagnostics.state != RuntimeState::PlacePreflight);

    std::vector<RuntimeState> collapsed{
        RuntimeState::Carry,
        RuntimeState::PlacePreflight,
    };
    RuntimeState previous = RuntimeState::PlacePreflight;
    for (int update = 0; update < kMaximumUpdates; ++update) {
        if (output.diagnostics.state != previous) {
            collapsed.push_back(output.diagnostics.state);
            previous = output.diagnostics.state;
        }
        if (output.diagnostics.state == RuntimeState::PlacePreflight ||
            output.diagnostics.state == RuntimeState::PlaceAlign ||
            output.diagnostics.state == RuntimeState::PlaceReplay ||
            output.diagnostics.state == RuntimeState::PlaceRelease) {
            assert(output.owns_pose);
            assert(output.suppress_steering);
        }
        if (output.diagnostics.state == RuntimeState::PlacePreflight) {
            ++place_preflight_publications;
            assert(output.diagnostics.place.selection_id ==
                   first.candidate.selection_id);
        }
        if (output.diagnostics.state == RuntimeState::PlaceReplay) {
            assert(output.diagnostics.attached);
            assert(output.diagnostics.object_state == ObjectState::Held);
        }
        if (output.diagnostics.state == RuntimeState::PlaceRelease) {
            assert(!output.diagnostics.attached);
            assert(output.diagnostics.object_state == ObjectState::Free);
            assert(output.diagnostics.target.id == fixture.request.target.id);
            assert(output.diagnostics.target.generation ==
                   fixture.request.target.generation + 1U);
            const InteractionTarget* placed = fixture.registry.find(
                output.diagnostics.target);
            assert(placed != nullptr);
            assert(exact(placed->object_world, output.object_world));
            assert(exact(
                placed->table_world,
                before_surface.support_volume_world));
            assert(exact(
                placed->table_size,
                before_surface.support_volume_size));
        }
        if (output.diagnostics.state == RuntimeState::Locomotion) break;
        output = advance(runtime, fixture.locomotion);
    }
    if (collapsed.empty() || collapsed.back() != RuntimeState::Locomotion) {
        collapsed.push_back(output.diagnostics.state);
    }
    const std::vector<RuntimeState> expected = {
        RuntimeState::Carry,
        RuntimeState::PlacePreflight,
        RuntimeState::PlaceAlign,
        RuntimeState::PlaceReplay,
        RuntimeState::PlaceRelease,
        RuntimeState::Locomotion,
    };
    assert(collapsed == expected);
    assert(place_preflight_publications == 1);
    assert(output.diagnostics.result == ResultCode::Succeeded);
    assert(output.diagnostics.reason == Reason::None);
    assert(!output.owns_pose && !output.suppress_steering);
    const InteractionTarget released_before_preview =
        *fixture.registry.find(output.diagnostics.target);
    const PlacementSurface surface_before_preview =
        *fixture.surface_registry.find(fixture.surface);
    const RuntimeDiagnostics diagnostics_before_preview =
        runtime.diagnostics();
    const PlaceStagingPreview after_release = runtime.preview_place(
        fixture.surface, fixture.place_affordance_id);
    assert(!after_release.accepted);
    assert(after_release.reason == Reason::OutOfRange);
    assert(exact(runtime.diagnostics(), diagnostics_before_preview));
    assert(exact(
        *fixture.registry.find(output.diagnostics.target),
        released_before_preview));
    assert(exact(
        *fixture.surface_registry.find(fixture.surface),
        surface_before_preview));

    const TargetHandle placed_handle = output.diagnostics.target;
    const std::optional<TargetHandle> resolved =
        fixture.registry.resolve_single_target(
            fixture.locomotion.pose.positions[g1_skeleton::Simulation],
            5.0F);
    assert(resolved.has_value());
    assert(*resolved == placed_handle);
    const InteractionTarget* placed = fixture.registry.find(placed_handle);
    const GraspAffordance* placed_affordance =
        fixture.registry.find_affordance(
            placed_handle, fixture.request.affordance_id);
    assert(placed != nullptr && placed_affordance != nullptr);
    const QueryInput destination_query = query_input_for(
        fixture, *placed, *placed_affordance);
    assert(exact(
        destination_query.table_world,
        before_surface.support_volume_world));
    assert(exact(
        destination_query.table_size,
        before_surface.support_volume_size));
    assert(!exact(destination_query.table_world, before_target.table_world));

    const PickRequest repick{
        placed_handle,
        fixture.request.affordance_id,
        1002U,
    };
    output = runtime.update(interact_input(fixture.locomotion, repick));
    assert(output.diagnostics.state == RuntimeState::Preflight);
    output = advance(runtime, fixture.locomotion);
    assert(output.diagnostics.state == RuntimeState::Align);
    assert(output.diagnostics.target == placed_handle);
}

void test_runtime_validates_complete_place_and_ik_configuration() {
    using namespace interaction;
    const float infinity = std::numeric_limits<float>::infinity();
    const float orientation_cap = 0.436332313F;

    const auto rejects_place = [](const auto& mutate) {
        RuntimeConfig config{};
        mutate(config.place);
        assert_both_runtime_constructors_reject_without_mutation(config);
    };
    rejects_place([](PlaceControllerConfig& value) {
        value.timing.canonical_fps = 0.0F;
    });
    rejects_place([](PlaceControllerConfig& value) {
        value.timing.playback_speed = 0.0F;
    });
    rejects_place([](PlaceControllerConfig& value) {
        value.timing.entry_blend_seconds = 0.0F;
    });
    rejects_place([](PlaceControllerConfig& value) {
        value.timing.reversed_commit_seconds = 0.0F;
    });
    rejects_place([](PlaceControllerConfig& value) {
        value.timing.maximum_alignment_seconds = 0.0F;
    });
    rejects_place([](PlaceControllerConfig& value) {
        value.timing.maximum_alignment_seconds = std::nextafter(
            value.timing.entry_blend_seconds, 0.0F);
    });
    rejects_place([](PlaceControllerConfig& value) {
        value.timing.maximum_alignment_seconds = std::nextafter(
            value.timing.reversed_commit_seconds, 0.0F);
    });
    rejects_place([](PlaceControllerConfig& value) {
        value.match.maximum_entry_root_error_m = 0.0F;
    });
    rejects_place([](PlaceControllerConfig& value) {
        value.match.maximum_entry_root_error_m = std::nextafter(
            0.25F, std::numeric_limits<float>::infinity());
    });
    rejects_place([](PlaceControllerConfig& value) {
        value.match.maximum_entry_yaw_error_radians = 0.0F;
    });
    rejects_place([orientation_cap](PlaceControllerConfig& value) {
        value.match.maximum_entry_yaw_error_radians = std::nextafter(
            orientation_cap, std::numeric_limits<float>::infinity());
    });
    rejects_place([](PlaceControllerConfig& value) {
        value.release_position_m = 0.0F;
    });
    rejects_place([](PlaceControllerConfig& value) {
        value.release_position_m = std::nextafter(
            0.02F, std::numeric_limits<float>::infinity());
    });
    rejects_place([](PlaceControllerConfig& value) {
        value.release_orientation_radians = 0.0F;
    });
    rejects_place([orientation_cap](PlaceControllerConfig& value) {
        value.release_orientation_radians = std::nextafter(
            orientation_cap, std::numeric_limits<float>::infinity());
    });

    using IKMember = float IKConfig::*;
    const std::array<IKMember, 8> ik_fields = {
        &IKConfig::maximum_request_position_m,
        &IKConfig::maximum_request_orientation_radians,
        &IKConfig::accepted_position_m,
        &IKConfig::accepted_orientation_radians,
        &IKConfig::damping,
        &IKConfig::finite_difference_radians,
        &IKConfig::orientation_scale_m_per_radian,
        &IKConfig::maximum_step_radians,
    };
    for (size_t index = 0; index < ik_fields.size(); ++index) {
        RuntimeConfig config{};
        config.ik.*ik_fields[index] = index < 4U ? -0.01F : 0.0F;
        assert_both_runtime_constructors_reject_without_mutation(config);

        config = RuntimeConfig{};
        config.ik.*ik_fields[index] = infinity;
        assert_both_runtime_constructors_reject_without_mutation(config);
    }
    {
        RuntimeConfig config{};
        config.ik.maximum_iterations = -1;
        assert_both_runtime_constructors_reject_without_mutation(config);
    }

    const float position_cap = 0.12F;
    for (IKMember capped : {
             &IKConfig::maximum_request_position_m,
             &IKConfig::maximum_request_orientation_radians}) {
        RuntimeConfig config{};
        config.ik.*capped = std::nextafter(
            capped == &IKConfig::maximum_request_position_m
                ? position_cap
                : orientation_cap,
            infinity);
        assert_both_runtime_constructors_reject_without_mutation(config);
    }

    RuntimeConfig cap_config{};
    cap_config.ik.maximum_request_position_m = position_cap;
    cap_config.ik.maximum_request_orientation_radians = orientation_cap;
    assert_both_runtime_constructors_accept(cap_config);
    RuntimeConfig tighter_config{};
    tighter_config.ik.maximum_request_position_m = 0.10F;
    tighter_config.ik.maximum_request_orientation_radians = 0.40F;
    assert_both_runtime_constructors_accept(tighter_config);

    for (const RuntimeConfig& config : {cap_config, tighter_config}) {
        RuntimeFixture fixture = make_runtime_fixture();
        InteractionRuntime runtime(
            fixture.database,
            fixture.features,
            fixture.registry,
            fixture.surface_registry,
            fixture.place_library,
            config);
        enter_carry_before_first_update(runtime, fixture);
        const PlaceStagingPreview preview = runtime.preview_place(
            fixture.surface, fixture.place_affordance_id);
        assert(preview.accepted);
        assert(exact(preview.ik, config.ik));
        assert(preview.ik.maximum_request_position_m ==
               config.ik.maximum_request_position_m);
        assert(preview.ik.maximum_request_orientation_radians ==
               config.ik.maximum_request_orientation_radians);
    }
}

void test_runtime_forwards_nondefault_place_config_exactly() {
    using namespace interaction;
    RuntimeFixture fixture = make_runtime_fixture();
    RuntimeConfig config{};
    config.place.timing.playback_speed = 0.85F;
    config.place.timing.entry_blend_seconds = 0.20F;
    config.place.timing.reversed_commit_seconds = 0.40F;
    config.place.timing.maximum_alignment_seconds = 0.90F;
    config.place.match.maximum_entry_root_error_m = 0.20F;
    config.place.match.maximum_entry_yaw_error_radians = 0.30F;
    config.place.release_position_m = 0.015F;
    config.place.release_orientation_radians = 0.10F;

    InteractionRuntime runtime(
        fixture.database,
        fixture.features,
        fixture.registry,
        fixture.surface_registry,
        fixture.place_library,
        config);
    enter_carry_before_first_update(runtime, fixture);
    const PlaceStagingPreview preview = runtime.preview_place(
        fixture.surface, fixture.place_affordance_id);
    assert(preview.accepted && preview.ready);
    assert(preview.candidate.timing.canonical_fps ==
           config.place.timing.canonical_fps);
    assert(preview.candidate.timing.playback_speed ==
           config.place.timing.playback_speed);
    assert(preview.candidate.timing.entry_blend_seconds ==
           config.place.timing.entry_blend_seconds);
    assert(preview.candidate.timing.reversed_commit_seconds ==
           config.place.timing.reversed_commit_seconds);
    assert(preview.candidate.timing.maximum_alignment_seconds ==
           config.place.timing.maximum_alignment_seconds);
    assert(preview.candidate.match.maximum_entry_root_error_m ==
           config.place.match.maximum_entry_root_error_m);
    assert(preview.candidate.match.maximum_entry_yaw_error_radians ==
           config.place.match.maximum_entry_yaw_error_radians);

    RuntimeOutput output = enter_place_align(
        runtime, fixture, place_request_for(runtime, fixture));
    assert(output.diagnostics.place.preflight_config_identity);
    assert(exact(output.diagnostics.place.preview.candidate, preview.candidate));
    const int32_t commit = preview.candidate.commit_frame;
    double previous_source = output.diagnostics.place.source_frame_exact;
    int ticks_to_commit = 0;
    while (output.diagnostics.state == RuntimeState::PlaceAlign) {
        output = advance(runtime, fixture.locomotion);
        ++ticks_to_commit;
        assert(output.diagnostics.place.source_frame_exact >= previous_source);
        previous_source = output.diagnostics.place.source_frame_exact;
    }
    assert(output.diagnostics.state == RuntimeState::PlaceReplay);
    assert(output.diagnostics.place.source_frame_exact >= commit);
    assert(ticks_to_commit == static_cast<int>(std::ceil(
        static_cast<double>(commit - preview.candidate.entry_frame) /
        config.place.timing.playback_speed)));

    RecordedPlaceClip& clip = fixture.place_library.recorded.front();
    clip.poses.at(static_cast<size_t>(clip.release_frame))
        .positions[kRightHandBone].x += 0.016F;
    RuntimeOutput last_safe = output;
    for (int update = 0; update < kMaximumUpdates; ++update) {
        output = advance(runtime, fixture.locomotion);
        if (output.diagnostics.state == RuntimeState::Carry) break;
        last_safe = output;
    }
    assert(output.diagnostics.state == RuntimeState::Carry);
    assert(output.diagnostics.result == ResultCode::Failed);
    assert(output.diagnostics.reason == Reason::ReleasePosition);
    assert(output.diagnostics.hand_position_error_m >
           config.place.release_position_m);
    assert(output.diagnostics.hand_position_error_m < 0.02F);
    assert(output.diagnostics.attached);
    assert(exact(output.pose, last_safe.pose));
    assert(exact(output.object_world, last_safe.object_world));
    assert_fresh_carry_next_sample(runtime, fixture, output, config);
}

void test_runtime_ik_identity_binds_every_scalar_and_iteration() {
    using namespace interaction;
    const auto preview_for = [](RuntimeConfig config) {
        RuntimeFixture fixture = make_runtime_fixture();
        InteractionRuntime runtime(
            fixture.database,
            fixture.features,
            fixture.registry,
            fixture.surface_registry,
            fixture.place_library,
            config);
        enter_carry_before_first_update(runtime, fixture);
        const PlaceStagingPreview preview = runtime.preview_place(
            fixture.surface, fixture.place_affordance_id);
        assert(preview.accepted && preview.ready);
        assert(exact(preview.ik, config.ik));
        assert(preview.ik_config_fingerprint != 0U);
        return preview;
    };

    const RuntimeConfig baseline_config{};
    const PlaceStagingPreview baseline = preview_for(baseline_config);
    const PlaceStagingPreview duplicate = preview_for(baseline_config);
    assert(baseline.ik_config_fingerprint ==
           duplicate.ik_config_fingerprint);
    assert(baseline.candidate.selection_id ==
           duplicate.candidate.selection_id);

    using IKMember = float IKConfig::*;
    const std::array<IKMember, 8> fields = {
        &IKConfig::maximum_request_position_m,
        &IKConfig::maximum_request_orientation_radians,
        &IKConfig::accepted_position_m,
        &IKConfig::accepted_orientation_radians,
        &IKConfig::damping,
        &IKConfig::finite_difference_radians,
        &IKConfig::orientation_scale_m_per_radian,
        &IKConfig::maximum_step_radians,
    };
    for (size_t index = 0; index < fields.size(); ++index) {
        RuntimeConfig perturbed = baseline_config;
        float& scalar = perturbed.ik.*fields[index];
        scalar = std::nextafter(
            scalar,
            index < 2U
                ? 0.0F
                : std::numeric_limits<float>::infinity());
        const PlaceStagingPreview changed = preview_for(perturbed);
        assert(changed.ik_config_fingerprint !=
               baseline.ik_config_fingerprint);
        assert(changed.candidate.selection_id !=
               baseline.candidate.selection_id);
    }
    RuntimeConfig iterated = baseline_config;
    ++iterated.ik.maximum_iterations;
    const PlaceStagingPreview iteration_changed = preview_for(iterated);
    assert(iteration_changed.ik_config_fingerprint !=
           baseline.ik_config_fingerprint);
    assert(iteration_changed.candidate.selection_id !=
           baseline.candidate.selection_id);

    RuntimeFixture fixture_a = make_runtime_fixture();
    RuntimeFixture fixture_b = make_runtime_fixture();
    RuntimeConfig config_b{};
    config_b.ik.damping = std::nextafter(
        config_b.ik.damping, std::numeric_limits<float>::infinity());
    InteractionRuntime runtime_a(
        fixture_a.database,
        fixture_a.features,
        fixture_a.registry,
        fixture_a.surface_registry,
        fixture_a.place_library,
        RuntimeConfig{});
    InteractionRuntime runtime_b(
        fixture_b.database,
        fixture_b.features,
        fixture_b.registry,
        fixture_b.surface_registry,
        fixture_b.place_library,
        config_b);
    enter_carry_before_first_update(runtime_a, fixture_a);
    enter_carry_before_first_update(runtime_b, fixture_b);
    const PlaceStagingPreview preview_a = runtime_a.preview_place(
        fixture_a.surface, fixture_a.place_affordance_id);
    const PlaceStagingPreview preview_b = runtime_b.preview_place(
        fixture_b.surface, fixture_b.place_affordance_id);
    assert(preview_a.candidate.selection_id !=
           preview_b.candidate.selection_id);
    PlaceRequest stale = place_request_for(runtime_b, fixture_b);
    stale.selection_id = preview_a.candidate.selection_id;
    assert_preflight_rejection_preserves_carry(
        runtime_b,
        fixture_b,
        place_interact_input(fixture_b.locomotion, stale),
        Reason::TargetChanged);
}

void test_place_preflight_rejections_preserve_frozen_carry() {
    using namespace interaction;
    const auto make_enabled = [](RuntimeFixture& fixture) {
        return InteractionRuntime(
            fixture.database,
            fixture.features,
            fixture.registry,
            fixture.surface_registry,
            fixture.place_library,
            RuntimeConfig{});
    };

    {
        RuntimeFixture fixture = make_runtime_fixture();
        InteractionRuntime runtime = make_enabled(fixture);
        enter_carry_before_first_update(runtime, fixture);
        assert_preflight_rejection_preserves_carry(
            runtime,
            fixture,
            place_interact_input(fixture.locomotion, std::nullopt),
            Reason::TargetUnavailable);
    }
    {
        RuntimeFixture fixture = make_runtime_fixture();
        InteractionRuntime runtime = make_enabled(fixture);
        enter_carry_before_first_update(runtime, fixture);
        PlaceRequest wrong = place_request_for(runtime, fixture);
        ++wrong.held_target.id;
        assert_preflight_rejection_preserves_carry(
            runtime,
            fixture,
            place_interact_input(fixture.locomotion, wrong),
            Reason::TargetChanged);
    }
    {
        RuntimeFixture fixture = make_runtime_fixture();
        InteractionRuntime runtime = make_enabled(fixture);
        enter_carry_before_first_update(runtime, fixture);
        PlaceRequest stale = place_request_for(runtime, fixture);
        ++stale.selection_id;
        assert_preflight_rejection_preserves_carry(
            runtime,
            fixture,
            place_interact_input(fixture.locomotion, stale),
            Reason::TargetChanged);
    }
    {
        RuntimeFixture fixture = make_runtime_fixture();
        InteractionRuntime runtime = make_enabled(fixture);
        enter_carry_before_first_update(runtime, fixture);
        PlaceRequest missing_action_id = place_request_for(runtime, fixture);
        missing_action_id.request_id = 0U;
        assert_preflight_rejection_preserves_carry(
            runtime,
            fixture,
            place_interact_input(fixture.locomotion, missing_action_id),
            Reason::TargetUnavailable);
    }
    {
        RuntimeFixture fixture = make_runtime_fixture();
        fixture.surface = fixture.surface_registry.upsert(
            runtime_fixture_detail::make_placement_surface(
                900U, vec3(0.55F, 0.65F, 3.0F)));
        InteractionRuntime runtime = make_enabled(fixture);
        enter_carry_before_first_update(runtime, fixture);
        const PlaceStagingPreview far = runtime.preview_place(
            fixture.surface, fixture.place_affordance_id);
        assert(far.accepted);
        assert(!far.ready);
        assert(far.root_error_m > 0.25F);
        PlaceRequest request{};
        request.held_target = fixture.request.target;
        request.surface = fixture.surface;
        request.affordance_id = fixture.place_affordance_id;
        request.request_id = 1001U;
        request.selection_id = far.candidate.selection_id;
        assert_preflight_rejection_preserves_carry(
            runtime,
            fixture,
            place_interact_input(fixture.locomotion, request),
            Reason::CorrectionLimit);
    }
    {
        RuntimeFixture fixture = make_runtime_fixture();
        fixture.surface = fixture.surface_registry.upsert(
            runtime_fixture_detail::make_placement_surface(
                900U, vec3(0.55F, 0.65F, 3.0F)));
        InteractionRuntime runtime = make_enabled(fixture);
        enter_carry_before_first_update(runtime, fixture);
        const PlaceStagingPreview far_snapshot = runtime.preview_place(
            fixture.surface, fixture.place_affordance_id);
        assert(far_snapshot.accepted && !far_snapshot.ready);
        advance(runtime, fixture.locomotion);
        const PlaceStagingPreview moved_snapshot = runtime.preview_place(
            fixture.surface, fixture.place_affordance_id);
        assert(moved_snapshot.accepted);
        assert(moved_snapshot.candidate.selection_id !=
               far_snapshot.candidate.selection_id);
        PlaceRequest stale_far{};
        stale_far.held_target = fixture.request.target;
        stale_far.surface = fixture.surface;
        stale_far.affordance_id = fixture.place_affordance_id;
        stale_far.request_id = 1001U;
        stale_far.selection_id = far_snapshot.candidate.selection_id;
        assert_preflight_rejection_preserves_carry(
            runtime,
            fixture,
            place_interact_input(fixture.locomotion, stale_far),
            Reason::TargetChanged);
    }
    {
        RuntimeFixture fixture = make_runtime_fixture();
        InteractionRuntime runtime(
            fixture.database,
            fixture.features,
            fixture.registry,
            RuntimeConfig{});
        enter_carry_before_first_update(runtime, fixture);
        const RuntimeDiagnostics before = runtime.diagnostics();
        const InteractionTarget target_before =
            *fixture.registry.find(fixture.request.target);
        const PlacementSurface surface_before =
            *fixture.surface_registry.find(fixture.surface);
        const PlaceStagingPreview unavailable = runtime.preview_place(
            fixture.surface, fixture.place_affordance_id);
        assert(!unavailable.accepted);
        assert(unavailable.reason == Reason::PackUnavailable);
        assert(exact(runtime.diagnostics(), before));
        assert(exact(
            *fixture.registry.find(fixture.request.target), target_before));
        assert(exact(
            *fixture.surface_registry.find(fixture.surface), surface_before));
        PlaceRequest request{};
        request.held_target = fixture.request.target;
        request.surface = fixture.surface;
        request.affordance_id = fixture.place_affordance_id;
        request.request_id = 1001U;
        request.selection_id = 1U;
        assert_preflight_rejection_preserves_carry(
            runtime,
            fixture,
            place_interact_input(fixture.locomotion, request),
            Reason::PackUnavailable);
    }
    {
        RuntimeFixture fixture = make_runtime_fixture();
        InteractionRuntime runtime = make_enabled(fixture);
        const RuntimeOutput carry = enter_carry_before_first_update(
            runtime, fixture);
        const int32_t pickup_clip = carry.diagnostics.clip;
        assert(pickup_clip >= 0);
        fixture.place_library.recorded.clear();
        fixture.database.active_hands.at(
            static_cast<size_t>(pickup_clip)) = 0U;
        const PlaceStagingPreview rejected = runtime.preview_place(
            fixture.surface, fixture.place_affordance_id);
        assert(!rejected.accepted);
        assert(rejected.reason == Reason::NoCandidate);
        PlaceRequest request{};
        request.held_target = fixture.request.target;
        request.surface = fixture.surface;
        request.affordance_id = fixture.place_affordance_id;
        request.request_id = 1001U;
        assert_preflight_rejection_preserves_carry(
            runtime,
            fixture,
            place_interact_input(fixture.locomotion, request),
            Reason::NoCandidate);
    }

    const auto assert_mutated_snapshot_rejects = [](
        RuntimeFixture& fixture,
        InteractionRuntime& runtime,
        bool mutate_target,
        bool mutate_library,
        bool mutate_surface,
        Reason expected_reason) {
        InteractionRuntime paused_control = runtime;
        const PlaceRequest request = place_request_for(runtime, fixture);
        RuntimeOutput output = runtime.update(place_interact_input(
            fixture.locomotion, request));
        assert(output.diagnostics.state == RuntimeState::PlacePreflight);
        const Pose frozen_pose = output.pose;
        const Transform frozen_object = output.object_world;
        if (mutate_target) {
            InteractionTarget* held = fixture.registry.find(
                fixture.request.target);
            assert(held != nullptr);
            held->object_dimensions.x = std::nextafter(
                held->object_dimensions.x,
                std::numeric_limits<float>::infinity());
        }
        if (mutate_library) {
            fixture.place_library.recorded.front()
                .object_poses.front().position.x += 0.001F;
        }
        if (mutate_surface) {
            PlacementSurface replacement =
                runtime_fixture_detail::make_placement_surface();
            replacement.surface_world.position.x += 0.01F;
            replacement.support_volume_world.position.x += 0.01F;
            fixture.surface_registry.upsert(replacement);
        }
        output = advance(runtime, fixture.locomotion);
        assert(output.diagnostics.state == RuntimeState::Carry);
        assert(output.diagnostics.result == ResultCode::Rejected);
        assert(output.diagnostics.reason == expected_reason);
        assert(output.diagnostics.attached);
        assert(exact(output.pose, frozen_pose));
        assert(exact(output.object_world, frozen_object));
        assert_held_by_original_owner(fixture);
        const RuntimeOutput trial_next = advance(runtime, fixture.locomotion);
        const RuntimeOutput control_next = advance(
            paused_control, fixture.locomotion);
        assert(exact(trial_next.pose, control_next.pose));
        assert(exact(trial_next.object_world, control_next.object_world));
    };
    {
        RuntimeFixture fixture = make_runtime_fixture();
        InteractionRuntime runtime = make_enabled(fixture);
        enter_carry_before_first_update(runtime, fixture);
        assert_mutated_snapshot_rejects(
            fixture, runtime, true, false, false, Reason::TargetChanged);
    }
    {
        RuntimeFixture fixture = make_runtime_fixture();
        InteractionRuntime runtime = make_enabled(fixture);
        enter_carry_before_first_update(runtime, fixture);
        assert_mutated_snapshot_rejects(
            fixture, runtime, false, true, false, Reason::TargetChanged);
    }
    {
        RuntimeFixture fixture = make_runtime_fixture();
        InteractionRuntime runtime = make_enabled(fixture);
        enter_carry_before_first_update(runtime, fixture);
        assert_mutated_snapshot_rejects(
            fixture, runtime, false, false, true, Reason::SurfaceChanged);
    }
}

void test_place_preflight_rejects_authored_slot_metadata_change() {
    using namespace interaction;

    RuntimeFixture fixture = make_runtime_fixture();
    seed_authored_interaction_slots(fixture);
    InteractionRuntime runtime(
        fixture.database,
        fixture.features,
        fixture.registry,
        fixture.surface_registry,
        fixture.place_library,
        RuntimeConfig{});
    RuntimeOutput output = enter_carry_before_first_update(runtime, fixture);
    const PlaceRequest request = place_request_for(runtime, fixture);
    output = runtime.update(place_interact_input(fixture.locomotion, request));
    assert(output.diagnostics.state == RuntimeState::PlacePreflight);

    InteractionTarget* held = fixture.registry.find(fixture.request.target);
    assert(held != nullptr);
    held->affordances.front()
        .interaction_slots[0].root_x_object_m += 0.001F;

    output = advance(runtime, fixture.locomotion);
    assert(output.diagnostics.state == RuntimeState::Carry);
    assert(output.diagnostics.result == ResultCode::Rejected);
    assert(output.diagnostics.reason == Reason::TargetChanged);
    assert(output.diagnostics.attached);
}

void test_carry_rejects_authored_slot_order_change() {
    using namespace interaction;

    RuntimeFixture fixture = make_runtime_fixture();
    seed_authored_interaction_slots(fixture);
    InteractionRuntime runtime(
        fixture.database, fixture.features, fixture.registry, RuntimeConfig{});
    RuntimeOutput output = enter_carry_before_first_update(runtime, fixture);
    assert(output.diagnostics.attached);

    InteractionTarget* held = fixture.registry.find(fixture.request.target);
    assert(held != nullptr);
    std::swap(
        held->affordances.front().interaction_slots[0],
        held->affordances.front().interaction_slots[1]);

    output = advance(runtime, fixture.locomotion);
    assert(output.diagnostics.state == RuntimeState::Locomotion);
    assert(output.diagnostics.result == ResultCode::Failed);
    assert(output.diagnostics.reason == Reason::TargetChanged);
    assert(!output.diagnostics.attached);
}

void test_post_begin_place_failures_reconstruct_fresh_carry() {
    using namespace interaction;
    {
        RuntimeFixture fixture = make_runtime_fixture();
        InteractionRuntime runtime(
            fixture.database,
            fixture.features,
            fixture.registry,
            fixture.surface_registry,
            fixture.place_library,
            RuntimeConfig{});
        enter_carry_before_first_update(runtime, fixture);
        RuntimeOutput output = enter_ready_place_align(runtime, fixture);
        for (int update = 0; update < 4; ++update) {
            output = advance(runtime, fixture.locomotion);
            assert(output.diagnostics.state == RuntimeState::PlaceAlign);
        }
        InteractionRuntime uncancelled = runtime;
        output = runtime.update(cancel_input(fixture.locomotion));
        const RuntimeOutput corresponding = advance(
            uncancelled, fixture.locomotion);
        assert(output.diagnostics.state == RuntimeState::Carry);
        assert(output.diagnostics.result == ResultCode::Cancelled);
        assert(output.diagnostics.reason == Reason::Cancelled);
        assert(output.diagnostics.attached);
        assert(exact(output.pose, corresponding.pose));
        assert(exact(output.object_world, corresponding.object_world));
        assert_fresh_carry_next_sample(runtime, fixture, output);
    }
    {
        RuntimeFixture fixture = make_runtime_fixture();
        InteractionRuntime runtime(
            fixture.database,
            fixture.features,
            fixture.registry,
            fixture.surface_registry,
            fixture.place_library,
            RuntimeConfig{});
        enter_carry_before_first_update(runtime, fixture);
        RuntimeOutput output = enter_ready_place_align(runtime, fixture);
        while (output.diagnostics.state == RuntimeState::PlaceAlign) {
            output = advance(runtime, fixture.locomotion);
        }
        assert(output.diagnostics.state == RuntimeState::PlaceReplay);
        assert(output.diagnostics.place.committed);
        fixture.place_library.recorded.front()
            .poses.at(static_cast<size_t>(
                fixture.place_library.recorded.front().release_frame))
            .positions[kRightHandBone].x += 0.021F;
        RuntimeOutput last_safe = output;
        for (int update = 0; update < kMaximumUpdates; ++update) {
            output = advance(runtime, fixture.locomotion);
            if (output.diagnostics.state == RuntimeState::Carry) break;
            last_safe = output;
        }
        assert(output.diagnostics.state == RuntimeState::Carry);
        assert(output.diagnostics.result == ResultCode::Failed);
        assert(output.diagnostics.reason == Reason::ReleasePosition);
        assert(output.diagnostics.hand_position_error_m > 0.02F);
        assert(output.diagnostics.attached);
        assert(output.diagnostics.object_state == ObjectState::Held);
        assert(exact(output.pose, last_safe.pose));
        assert(exact(output.object_world, last_safe.object_world));
        assert_held_by_original_owner(fixture);
        assert_fresh_carry_next_sample(runtime, fixture, output);
    }
    {
        RuntimeFixture trial_fixture = make_runtime_fixture();
        RuntimeFixture control_fixture = make_runtime_fixture();
        InteractionRuntime trial(
            trial_fixture.database,
            trial_fixture.features,
            trial_fixture.registry,
            trial_fixture.surface_registry,
            trial_fixture.place_library,
            RuntimeConfig{});
        InteractionRuntime control(
            control_fixture.database,
            control_fixture.features,
            control_fixture.registry,
            control_fixture.surface_registry,
            control_fixture.place_library,
            RuntimeConfig{});
        enter_carry_before_first_update(trial, trial_fixture);
        enter_carry_before_first_update(control, control_fixture);
        RuntimeOutput trial_output = enter_ready_place_align(
            trial, trial_fixture);
        RuntimeOutput control_output = enter_ready_place_align(
            control, control_fixture);
        assert(exact(trial_output, control_output));

        PlacementSurface replacement =
            runtime_fixture_detail::make_placement_surface();
        replacement.surface_world.position.x += 0.01F;
        replacement.support_volume_world.position.x += 0.01F;
        const SurfaceHandle replacement_handle =
            trial_fixture.surface_registry.upsert(replacement);
        assert(replacement_handle.generation ==
               trial_fixture.surface.generation + 1U);
        const InteractionTarget held_before =
            *trial_fixture.registry.find(trial_fixture.request.target);
        for (int update = 0; update < kMaximumUpdates; ++update) {
            trial_output = advance(trial, trial_fixture.locomotion);
            control_output = advance(control, control_fixture.locomotion);
            if (trial_output.diagnostics.state == RuntimeState::Carry) break;
            assert(exact(trial_output.pose, control_output.pose));
            assert(exact(trial_output.object_world, control_output.object_world));
        }
        assert(trial_output.diagnostics.state == RuntimeState::Carry);
        assert(trial_output.diagnostics.result == ResultCode::Failed);
        assert(trial_output.diagnostics.reason == Reason::SurfaceChanged);
        assert(trial_output.diagnostics.attached);
        assert(control_output.diagnostics.state == RuntimeState::PlaceRelease);
        assert(exact(trial_output.pose, control_output.pose));
        assert(exact(trial_output.object_world, control_output.object_world));
        assert(exact(
            *trial_fixture.registry.find(trial_fixture.request.target),
            held_before));
        assert_fresh_carry_next_sample(
            trial, trial_fixture, trial_output);
    }
    {
        RuntimeFixture fixture = make_runtime_fixture();
        fixture.registry = TargetRegistry{};
        InteractionTarget target = runtime_fixture_detail::make_target();
        target.handle.generation = std::numeric_limits<uint32_t>::max();
        const TargetHandle maximum = fixture.registry.upsert(target);
        assert(maximum.generation == std::numeric_limits<uint32_t>::max());
        fixture.request = {
            maximum,
            runtime_fixture_detail::kAffordanceId,
            runtime_fixture_detail::kRequestId,
        };
        InteractionRuntime runtime(
            fixture.database,
            fixture.features,
            fixture.registry,
            fixture.surface_registry,
            fixture.place_library,
            RuntimeConfig{});
        enter_carry_before_first_update(runtime, fixture);
        RuntimeOutput output = enter_ready_place_align(runtime, fixture);
        const InteractionTarget held_before =
            *fixture.registry.find(fixture.request.target);
        for (int update = 0; update < kMaximumUpdates; ++update) {
            output = advance(runtime, fixture.locomotion);
            if (output.diagnostics.state == RuntimeState::Carry) break;
        }
        assert(output.diagnostics.state == RuntimeState::Carry);
        assert(output.diagnostics.result == ResultCode::Failed);
        assert(output.diagnostics.reason == Reason::TargetChanged);
        assert(output.diagnostics.attached);
        assert(output.diagnostics.target == maximum);
        assert(exact(
            *fixture.registry.find(fixture.request.target), held_before));
        assert(exact(
            fixture.registry.find(fixture.request.target)->table_world,
            held_before.table_world));
        assert(exact(
            fixture.registry.find(fixture.request.target)->table_size,
            held_before.table_size));
        assert_fresh_carry_next_sample(runtime, fixture, output);
    }
}

void test_place_cancellation_boundaries_for_recorded_and_reverse() {
    using namespace interaction;
    for (bool reverse : {false, true}) {
        for (int relative_to_commit : {-1, 0, 1}) {
            RuntimeFixture trial_fixture = reverse
                ? reverse_place_runtime_fixture()
                : make_runtime_fixture();
            RuntimeFixture control_fixture = reverse
                ? reverse_place_runtime_fixture()
                : make_runtime_fixture();
            RuntimeConfig runtime_config{};
            if (reverse) {
                runtime_config.ik.accepted_position_m =
                    runtime_config.ik.maximum_request_position_m;
                runtime_config.ik.accepted_orientation_radians =
                    runtime_config.ik.maximum_request_orientation_radians;
            }
            InteractionRuntime trial(
                trial_fixture.database,
                trial_fixture.features,
                trial_fixture.registry,
                trial_fixture.surface_registry,
                trial_fixture.place_library,
                runtime_config);
            InteractionRuntime control(
                control_fixture.database,
                control_fixture.features,
                control_fixture.registry,
                control_fixture.surface_registry,
                control_fixture.place_library,
                runtime_config);
            enter_carry_before_first_update(trial, trial_fixture);
            enter_carry_before_first_update(control, control_fixture);
            const PlaceStagingPreview preview = trial.preview_place(
                trial_fixture.surface,
                trial_fixture.place_affordance_id);
            assert(preview.accepted && preview.ready);
            assert(preview.candidate.mode ==
                   (reverse
                        ? PlaceMotionMode::ReversedPickup
                        : PlaceMotionMode::RecordedPlace));
            assert(preview.candidate.direction == (reverse ? -1 : 1));
            RuntimeOutput trial_output = enter_ready_place_align(
                trial, trial_fixture);
            RuntimeOutput control_output = enter_ready_place_align(
                control, control_fixture);
            assert(exact(trial_output, control_output));

            const int32_t direction = preview.candidate.direction;
            const double target_source =
                preview.candidate.commit_frame +
                relative_to_commit * direction;
            const double source_before_cancel = target_source - direction;
            for (int update = 0; update < kMaximumUpdates; ++update) {
                if (trial_output.diagnostics.place.source_frame_exact ==
                    source_before_cancel) {
                    break;
                }
                if (direction > 0) {
                    assert(
                        trial_output.diagnostics.place.source_frame_exact <
                        source_before_cancel);
                } else {
                    assert(
                        trial_output.diagnostics.place.source_frame_exact >
                        source_before_cancel);
                }
                trial_output = advance(trial, trial_fixture.locomotion);
                control_output = advance(control, control_fixture.locomotion);
                assert(exact(trial_output, control_output));
            }
            assert(trial_output.diagnostics.place.source_frame_exact ==
                   source_before_cancel);

            trial_output = trial.update(cancel_input(
                trial_fixture.locomotion));
            control_output = advance(control, control_fixture.locomotion);
            assert(control_output.diagnostics.place.source_frame_exact ==
                   target_source);
            if (relative_to_commit < 0) {
                assert(trial_output.diagnostics.state == RuntimeState::Carry);
                assert(trial_output.diagnostics.result ==
                       ResultCode::Cancelled);
                assert(trial_output.diagnostics.reason == Reason::Cancelled);
                assert(trial_output.diagnostics.attached);
                assert(control_output.diagnostics.state ==
                       RuntimeState::PlaceAlign);
                assert(exact(trial_output.pose, control_output.pose));
                assert(exact(
                    trial_output.object_world,
                    control_output.object_world));
                assert_fresh_carry_next_sample(
                    trial, trial_fixture, trial_output, runtime_config);
            } else {
                assert(exact(trial_output, control_output));
                assert(trial_output.diagnostics.state ==
                       RuntimeState::PlaceReplay);
                assert(trial_output.diagnostics.attached);
                assert(trial_output.diagnostics.object_state ==
                       ObjectState::Held);
                assert(trial_output.diagnostics.place.committed);
                assert(trial_output.diagnostics.reason == Reason::None);
            }
        }
    }
}

void test_cancel_on_release_update_preserves_release_for_both_modes() {
    using namespace interaction;
    for (bool reverse : {false, true}) {
        RuntimeFixture trial_fixture = reverse
            ? reverse_place_runtime_fixture()
            : make_runtime_fixture();
        RuntimeFixture control_fixture = reverse
            ? reverse_place_runtime_fixture()
            : make_runtime_fixture();
        RuntimeConfig runtime_config{};
        if (reverse) {
            runtime_config.ik.accepted_position_m =
                runtime_config.ik.maximum_request_position_m;
            runtime_config.ik.accepted_orientation_radians =
                runtime_config.ik.maximum_request_orientation_radians;
        }
        InteractionRuntime trial(
            trial_fixture.database,
            trial_fixture.features,
            trial_fixture.registry,
            trial_fixture.surface_registry,
            trial_fixture.place_library,
            runtime_config);
        InteractionRuntime control(
            control_fixture.database,
            control_fixture.features,
            control_fixture.registry,
            control_fixture.surface_registry,
            control_fixture.place_library,
            runtime_config);
        enter_carry_before_first_update(trial, trial_fixture);
        enter_carry_before_first_update(control, control_fixture);
        const PlaceStagingPreview preview = trial.preview_place(
            trial_fixture.surface,
            trial_fixture.place_affordance_id);
        assert(preview.accepted && preview.ready);
        assert(preview.candidate.mode ==
               (reverse
                    ? PlaceMotionMode::ReversedPickup
                    : PlaceMotionMode::RecordedPlace));
        RuntimeOutput trial_output = enter_ready_place_align(
            trial, trial_fixture, reverse ? 4001U : 4002U);
        RuntimeOutput control_output = enter_ready_place_align(
            control, control_fixture, reverse ? 4001U : 4002U);
        assert(exact(trial_output, control_output));

        const int32_t direction = preview.candidate.direction;
        const double immediately_before_release =
            preview.candidate.release_frame - direction;
        for (int update = 0; update < kMaximumUpdates; ++update) {
            if (trial_output.diagnostics.place.source_frame_exact ==
                immediately_before_release) {
                break;
            }
            if (direction > 0) {
                assert(
                    trial_output.diagnostics.place.source_frame_exact <
                    immediately_before_release);
            } else {
                assert(
                    trial_output.diagnostics.place.source_frame_exact >
                    immediately_before_release);
            }
            trial_output = advance(trial, trial_fixture.locomotion);
            control_output = advance(control, control_fixture.locomotion);
            assert(exact(trial_output, control_output));
        }
        assert(trial_output.diagnostics.state == RuntimeState::PlaceReplay);
        assert(trial_output.diagnostics.attached);
        assert(trial_output.diagnostics.place.source_frame_exact ==
               immediately_before_release);

        const PlacementSurface destination =
            *trial_fixture.surface_registry.find(trial_fixture.surface);
        trial_output = trial.update(cancel_input(trial_fixture.locomotion));
        control_output = advance(control, control_fixture.locomotion);
        assert(exact(trial_output, control_output));
        assert(trial_output.diagnostics.state == RuntimeState::PlaceRelease);
        assert(trial_output.diagnostics.result == ResultCode::Succeeded);
        assert(trial_output.diagnostics.reason == Reason::None);
        assert(!trial_output.diagnostics.attached);
        assert(trial_output.diagnostics.object_state == ObjectState::Free);
        assert(trial_output.diagnostics.target.id ==
               trial_fixture.request.target.id);
        assert(trial_output.diagnostics.target.generation ==
               trial_fixture.request.target.generation + 1U);
        const TargetHandle placed_handle = trial_output.diagnostics.target;
        const InteractionTarget* trial_placed =
            trial_fixture.registry.find(placed_handle);
        const InteractionTarget* control_placed =
            control_fixture.registry.find(placed_handle);
        assert(trial_placed != nullptr && control_placed != nullptr);
        assert(exact(*trial_placed, *control_placed));
        assert(exact(trial_placed->object_world, trial_output.object_world));
        assert(exact(
            trial_placed->table_world,
            destination.support_volume_world));
        assert(exact(
            trial_placed->table_size,
            destination.support_volume_size));

        for (int update = 0; update < kMaximumUpdates; ++update) {
            if (trial_output.diagnostics.state == RuntimeState::Locomotion) {
                break;
            }
            trial_output = advance(trial, trial_fixture.locomotion);
            control_output = advance(control, control_fixture.locomotion);
            assert(exact(trial_output, control_output));
            assert(trial_fixture.registry.find(placed_handle) != nullptr);
            assert(control_fixture.registry.find(placed_handle) != nullptr);
        }
        assert(trial_output.diagnostics.state == RuntimeState::Locomotion);
        assert(trial_output.diagnostics.result == ResultCode::Succeeded);
        assert(trial_output.diagnostics.reason == Reason::None);
        assert(trial_output.diagnostics.target == placed_handle);
        assert(trial_fixture.registry.find(placed_handle)->handle ==
               placed_handle);
        assert(control_fixture.registry.find(placed_handle)->handle ==
               placed_handle);
    }
}

void test_reset_and_duplicate_edges_are_ignored_in_every_place_state() {
    using namespace interaction;
    for (bool reset : {false, true}) {
        RuntimeFixture trial_fixture = make_runtime_fixture();
        RuntimeFixture control_fixture = make_runtime_fixture();
        InteractionRuntime trial(
            trial_fixture.database,
            trial_fixture.features,
            trial_fixture.registry,
            trial_fixture.surface_registry,
            trial_fixture.place_library,
            RuntimeConfig{});
        InteractionRuntime control(
            control_fixture.database,
            control_fixture.features,
            control_fixture.registry,
            control_fixture.surface_registry,
            control_fixture.place_library,
            RuntimeConfig{});
        enter_carry_before_first_update(trial, trial_fixture);
        enter_carry_before_first_update(control, control_fixture);
        const PlaceRequest request = place_request_for(
            trial, trial_fixture, reset ? 2001U : 2002U);
        RuntimeOutput trial_output = trial.update(place_interact_input(
            trial_fixture.locomotion, request));
        RuntimeOutput control_output = control.update(place_interact_input(
            control_fixture.locomotion, request));
        assert(exact(trial_output, control_output));
        assert(trial_output.diagnostics.state ==
               RuntimeState::PlacePreflight);

        uint32_t observed_states = 0U;
        for (int update = 0; update < kMaximumUpdates; ++update) {
            switch (trial_output.diagnostics.state) {
                case RuntimeState::PlacePreflight:
                    observed_states |= 1U << 0U;
                    break;
                case RuntimeState::PlaceAlign:
                    observed_states |= 1U << 1U;
                    break;
                case RuntimeState::PlaceReplay:
                    observed_states |= 1U << 2U;
                    break;
                case RuntimeState::PlaceRelease:
                    observed_states |= 1U << 3U;
                    break;
                case RuntimeState::Locomotion:
                    break;
                default:
                    assert(false && "unexpected state in place equivalence");
            }
            if (trial_output.diagnostics.state == RuntimeState::Locomotion) {
                break;
            }

            RuntimeInput trial_input = idle_input(trial_fixture.locomotion);
            if (reset) {
                trial_input.reset_pressed = true;
            } else {
                trial_input.interact_pressed = true;
                trial_input.place_request = request;
            }
            trial_output = trial.update(trial_input);
            control_output = advance(control, control_fixture.locomotion);
            assert(exact(trial_output, control_output));
        }
        assert(observed_states == 0x0FU);
        assert(trial_output.diagnostics.state == RuntimeState::Locomotion);
        assert(trial_output.diagnostics.result == ResultCode::Succeeded);
        assert(trial_output.diagnostics.reason == Reason::None);
    }
}

void test_recorded_and_reverse_place_diagnostics_are_deterministic() {
    using namespace interaction;
    for (bool reverse : {false, true}) {
        RuntimeFixture first_fixture = reverse
            ? reverse_place_runtime_fixture()
            : make_runtime_fixture();
        RuntimeFixture second_fixture = reverse
            ? reverse_place_runtime_fixture()
            : make_runtime_fixture();
        RuntimeConfig config{};
        if (reverse) {
            config.ik.accepted_position_m =
                config.ik.maximum_request_position_m;
            config.ik.accepted_orientation_radians =
                config.ik.maximum_request_orientation_radians;
        }
        InteractionRuntime first(
            first_fixture.database,
            first_fixture.features,
            first_fixture.registry,
            first_fixture.surface_registry,
            first_fixture.place_library,
            config);
        InteractionRuntime second(
            second_fixture.database,
            second_fixture.features,
            second_fixture.registry,
            second_fixture.surface_registry,
            second_fixture.place_library,
            config);
        enter_carry_before_first_update(first, first_fixture);
        enter_carry_before_first_update(second, second_fixture);
        const PlaceStagingPreview preview = first.preview_place(
            first_fixture.surface, first_fixture.place_affordance_id);
        const PlaceStagingPreview duplicate = second.preview_place(
            second_fixture.surface, second_fixture.place_affordance_id);
        assert(preview.accepted && preview.ready);
        assert(exact(preview, duplicate));
        const PlaceMotionMode expected_mode = reverse
            ? PlaceMotionMode::ReversedPickup
            : PlaceMotionMode::RecordedPlace;
        assert(preview.candidate.mode == expected_mode);

        const PlaceRequest request = place_request_for(first, first_fixture);
        RuntimeOutput first_output = first.update(place_interact_input(
            first_fixture.locomotion, request));
        RuntimeOutput second_output = second.update(place_interact_input(
            second_fixture.locomotion, request));
        assert(exact(first_output, second_output));
        double previous_source =
            first_output.diagnostics.place.source_frame_exact;
        bool have_source = false;
        bool saw_align = false;
        bool saw_replay = false;
        bool saw_release = false;
        int stop_sample_count = 0;
        for (int update = 0; update < kMaximumUpdates; ++update) {
            const RuntimeState state = first_output.diagnostics.state;
            if (state == RuntimeState::Locomotion) break;
            assert(state == RuntimeState::PlacePreflight ||
                   state == RuntimeState::PlaceAlign ||
                   state == RuntimeState::PlaceReplay ||
                   state == RuntimeState::PlaceRelease);
            assert(first_output.owns_pose);
            assert(first_output.suppress_steering);
            assert(first_output.diagnostics.place.mode == expected_mode);
            assert(first_output.diagnostics.place.selection_id ==
                   preview.candidate.selection_id);
            assert(first_output.diagnostics.place.ik_config_fingerprint ==
                   preview.ik_config_fingerprint);
            assert(exact(
                first_output.diagnostics.place.effective_ik,
                config.ik));
            assert(first_output.diagnostics.place.candidate_certified);
            if (state != RuntimeState::PlacePreflight) {
                assert(first_output.diagnostics.place.preflight_config_identity);
            }
            for (float correction : {
                     first_output.diagnostics.place
                         .requested_root_correction_m,
                     first_output.diagnostics.place.applied_root_correction_m,
                     first_output.diagnostics.place
                         .requested_yaw_correction_radians,
                     first_output.diagnostics.place
                         .applied_yaw_correction_radians,
                     first_output.diagnostics.place
                         .requested_hand_correction_m,
                     first_output.diagnostics.place.applied_hand_correction_m,
                     first_output.diagnostics.place
                         .requested_hand_orientation_radians,
                     first_output.diagnostics.place
                         .applied_hand_orientation_radians}) {
                assert(std::isfinite(correction));
                assert(correction >= 0.0F);
            }
            if (state == RuntimeState::PlaceAlign ||
                state == RuntimeState::PlaceReplay ||
                state == RuntimeState::PlaceRelease) {
                const double source =
                    first_output.diagnostics.place.source_frame_exact;
                assert(std::isfinite(source));
                if (have_source) {
                    if (preview.candidate.direction > 0) {
                        assert(source >= previous_source);
                    } else {
                        assert(source <= previous_source);
                    }
                }
                previous_source = source;
                have_source = true;
                if (source == preview.candidate.stop_frame) {
                    ++stop_sample_count;
                }
            }
            saw_align = saw_align || state == RuntimeState::PlaceAlign;
            saw_replay = saw_replay || state == RuntimeState::PlaceReplay;
            saw_release = saw_release || state == RuntimeState::PlaceRelease;
            if (state == RuntimeState::PlaceRelease) {
                assert(first_output.diagnostics.place.released);
                assert(!first_output.diagnostics.attached);
                assert(first_output.diagnostics.place
                           .requested_root_correction_m == 0.0F);
                assert(first_output.diagnostics.place
                           .applied_root_correction_m == 0.0F);
                assert(first_output.diagnostics.place
                           .requested_yaw_correction_radians == 0.0F);
                assert(first_output.diagnostics.place
                           .applied_yaw_correction_radians == 0.0F);
                assert(first_output.diagnostics.place
                           .requested_hand_correction_m == 0.0F);
                assert(first_output.diagnostics.place
                           .applied_hand_correction_m == 0.0F);
                assert(first_output.diagnostics.place
                           .requested_hand_orientation_radians == 0.0F);
                assert(first_output.diagnostics.place
                           .applied_hand_orientation_radians == 0.0F);
            }
            first_output = advance(first, first_fixture.locomotion);
            second_output = advance(second, second_fixture.locomotion);
            assert(exact(first_output, second_output));
        }
        assert(first_output.diagnostics.state == RuntimeState::Locomotion);
        assert(first_output.diagnostics.result == ResultCode::Succeeded);
        assert(first_output.diagnostics.reason == Reason::None);
        assert(saw_align && saw_replay && saw_release);
        assert(stop_sample_count == 1);
    }
}

void test_disabled_is_permanent_pose_passthrough() {
    using namespace interaction;
    const RuntimeFixture fixture = make_runtime_fixture();
    InteractionRuntime runtime = InteractionRuntime::disabled(
        Reason::PackUnavailable);

    RuntimeOutput output = runtime.update(interact_input(
        fixture.locomotion, fixture.request));
    assert(runtime.state() == RuntimeState::Disabled);
    assert(output.diagnostics.state == RuntimeState::Disabled);
    assert(output.diagnostics.reason == Reason::PackUnavailable);
    assert(!output.diagnostics.pack_available);
    assert(!output.owns_pose && !output.suppress_steering);
    assert(exact(output.pose, fixture.locomotion.pose));

    output = runtime.update(reset_input(fixture.locomotion));
    assert(output.diagnostics.state == RuntimeState::Disabled);
    assert(output.diagnostics.reason == Reason::PackUnavailable);
    assert(!output.owns_pose && !output.suppress_steering);
    assert(exact(output.pose, fixture.locomotion.pose));

    for (float arbitrary_dt : {
             0.0F,
             -1.0F,
             std::numeric_limits<float>::quiet_NaN(),
             std::numeric_limits<float>::infinity()}) {
        RuntimeInput arbitrary = interact_input(
            fixture.locomotion, fixture.request);
        arbitrary.dt = arbitrary_dt;
        output = runtime.update(arbitrary);
        assert(output.diagnostics.state == RuntimeState::Disabled);
        assert(output.diagnostics.reason == Reason::PackUnavailable);
        assert(!output.owns_pose && !output.suppress_steering);
        assert(exact(output.pose, fixture.locomotion.pose));
    }
}

void test_enabled_runtime_rejects_noncanonical_dt_before_mutation() {
    using namespace interaction;
    const std::array<float, 6> invalid_steps = {
        0.0F,
        1.0F / 60.0F,
        0.12F,
        -kDt,
        std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::infinity(),
    };
    for (float invalid_dt : invalid_steps) {
        RuntimeFixture fixture = make_runtime_fixture();
        InteractionRuntime runtime(
            fixture.database,
            fixture.features,
            fixture.registry,
            RuntimeConfig{});
        const RuntimeDiagnostics before = runtime.diagnostics();
        RuntimeInput invalid = interact_input(
            fixture.locomotion, fixture.request);
        invalid.dt = invalid_dt;
        bool threw = false;
        try {
            (void)runtime.update(invalid);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        assert(threw);
        assert(runtime.state() == RuntimeState::Locomotion);
        assert(exact(runtime.diagnostics(), before));
        assert_free(fixture.registry, fixture.request.target);

        const RuntimeOutput retried = runtime.update(interact_input(
            fixture.locomotion, fixture.request));
        assert(retried.diagnostics.state == RuntimeState::Preflight);
        assert_free(fixture.registry, fixture.request.target);
    }
}

void test_noncanonical_dt_cannot_mutate_owned_runtime_state() {
    using namespace interaction;
    RuntimeFixture fixture = make_runtime_fixture();
    RuntimeFixture reference_fixture = make_runtime_fixture();
    InteractionRuntime runtime(
        fixture.database,
        fixture.features,
        fixture.registry,
        RuntimeConfig{});
    InteractionRuntime reference(
        reference_fixture.database,
        reference_fixture.features,
        reference_fixture.registry,
        RuntimeConfig{});
    const RuntimeOutput before = enter_align(runtime, fixture);
    const RuntimeOutput reference_before = enter_align(
        reference, reference_fixture);
    assert(exact(before.diagnostics, reference_before.diagnostics));
    const RuntimeDiagnostics diagnostics_before = runtime.diagnostics();
    const InteractionTarget* target_before = fixture.registry.find(
        fixture.request.target);
    assert(target_before != nullptr);
    const Transform object_before = target_before->object_world;

    RuntimeInput invalid = cancel_input(fixture.locomotion);
    invalid.dt = 0.12F;
    bool threw = false;
    try {
        (void)runtime.update(invalid);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
    assert(runtime.state() == RuntimeState::Align);
    assert(exact(runtime.diagnostics(), diagnostics_before));
    const InteractionTarget* target_after = fixture.registry.find(
        fixture.request.target);
    assert(target_after != nullptr);
    assert(target_after->state == ObjectState::Targeted);
    assert(target_after->owner_request == fixture.request.request_id);
    assert(exact(target_after->object_world, object_before));

    const RuntimeOutput after = advance(runtime, fixture.locomotion);
    const RuntimeOutput reference_after = advance(
        reference, reference_fixture.locomotion);
    assert(exact(after.diagnostics, reference_after.diagnostics));
    assert(exact(after.pose, reference_after.pose));
    assert(exact(after.object_world, reference_after.object_world));
}

void test_interact_without_explicit_request_is_rejected_after_preflight() {
    using namespace interaction;
    RuntimeFixture fixture = make_runtime_fixture();
    InteractionRuntime runtime(
        fixture.database, fixture.features, fixture.registry, RuntimeConfig{});

    RuntimeOutput output = runtime.update(interact_input(
        fixture.locomotion, std::nullopt));
    assert(output.diagnostics.state == RuntimeState::Preflight);
    assert(output.diagnostics.result == ResultCode::None);
    assert(!output.owns_pose && !output.suppress_steering);
    assert_free(fixture.registry, fixture.request.target);

    output = advance(runtime, fixture.locomotion);
    assert(output.diagnostics.state == RuntimeState::Locomotion);
    assert(output.diagnostics.result == ResultCode::Rejected);
    assert(output.diagnostics.reason == Reason::TargetUnavailable);
    assert(!output.diagnostics.attached);
    assert(!output.owns_pose && !output.suppress_steering);
    assert_free(fixture.registry, fixture.request.target);
}

void test_success_order_carry_and_reset() {
    using namespace interaction;
    RuntimeFixture fixture = make_runtime_fixture();
    RuntimeConfig config{};
    InteractionRuntime runtime(
        fixture.database, fixture.features, fixture.registry, config);
    assert(runtime.state() == RuntimeState::Locomotion);

    RuntimeOutput output = runtime.update(interact_input(
        fixture.locomotion, fixture.request));
    assert(output.diagnostics.state == RuntimeState::Preflight);
    assert(!output.owns_pose && !output.suppress_steering);
    assert_free(fixture.registry, fixture.request.target);

    output = advance(runtime, fixture.locomotion);
    assert(output.diagnostics.state == RuntimeState::Align);
    assert(output.diagnostics.result == ResultCode::Accepted);
    assert(output.diagnostics.reason == Reason::None);
    assert(output.diagnostics.target == fixture.request.target);
    assert(output.diagnostics.affordance_id == fixture.request.affordance_id);
    assert(output.diagnostics.object_state == ObjectState::Targeted);
    assert(output.diagnostics.pack_available);
    assert(output.owns_pose && !output.suppress_steering);
    const InteractionTarget* reserved = fixture.registry.find(
        fixture.request.target);
    assert(reserved != nullptr);
    assert(reserved->state == ObjectState::Targeted);
    assert(reserved->owner_request == fixture.request.request_id);

    output = advance_until(
        runtime,
        fixture.locomotion,
        RuntimeState::PickupReplay,
        RuntimeState::Align);
    assert(output.diagnostics.result == ResultCode::Accepted);
    assert(output.owns_pose && output.suppress_steering);

    output = advance_until(
        runtime,
        fixture.locomotion,
        RuntimeState::Hold,
        RuntimeState::PickupReplay);
    assert(output.diagnostics.attached);
    assert(output.diagnostics.object_state == ObjectState::Attached);
    assert(output.owns_pose && output.suppress_steering);

    output = advance_until(
        runtime,
        fixture.locomotion,
        RuntimeState::Carry,
        RuntimeState::Hold);
    assert(output.diagnostics.result == ResultCode::Succeeded);
    assert(output.diagnostics.reason == Reason::None);
    assert(output.diagnostics.attached);
    assert(output.diagnostics.object_state == ObjectState::Held);
    assert(output.owns_pose && !output.suppress_steering);

    const uint32_t generation_before_reset =
        runtime.diagnostics().target.generation;
    output = runtime.update(reset_input(fixture.locomotion));
    assert(output.diagnostics.state == RuntimeState::Locomotion);
    assert(output.diagnostics.result == ResultCode::Reset);
    assert(output.diagnostics.reason == Reason::Reset);
    assert(!output.diagnostics.attached);
    assert(!output.diagnostics.inactive_arm_tracks_locomotion);
    assert(!output.owns_pose && !output.suppress_steering);
    assert(exact(output.object_world, fixture.original_object_world));

    const std::optional<TargetHandle> restored =
        fixture.registry.resolve_single_target(vec3(0.0F, 0.0F, 3.0F), 1.0F);
    assert(restored.has_value());
    assert(restored->generation == generation_before_reset + 1U);
    const InteractionTarget* restored_target = fixture.registry.find(*restored);
    assert(restored_target != nullptr);
    assert(restored_target->state == ObjectState::Free);
    assert(restored_target->owner_request == 0U);
    assert(exact(restored_target->object_world, fixture.original_object_world));

    output = advance(runtime, fixture.locomotion);
    const std::optional<TargetHandle> after_idle =
        fixture.registry.resolve_single_target(vec3(0.0F, 0.0F, 3.0F), 1.0F);
    assert(after_idle == restored);
    assert(output.diagnostics.result == ResultCode::Reset);
}

void test_layered_carry_publishes_inactive_arm_authority_transactionally() {
    using namespace interaction;
    RuntimeFixture fixture = make_runtime_fixture();
    RuntimeConfig config{};
    config.carry.minimum_root_displacement_m = 10.0F;
    InteractionRuntime runtime(
        fixture.database, fixture.features, fixture.registry, config);

    RuntimeOutput output = enter_carry_before_first_update(runtime, fixture);
    assert(!output.diagnostics.inactive_arm_targets_locomotion);
    LocomotionSnapshot carry_locomotion = fixture.locomotion;
    carry_locomotion.pose = output.pose;
    for (int update = 0;
         update < 32 &&
         !output.diagnostics.inactive_arm_tracks_locomotion;
         ++update) {
        output = advance(runtime, carry_locomotion);
        if (!output.diagnostics.inactive_arm_tracks_locomotion) {
            assert(output.diagnostics.state == RuntimeState::Carry);
            assert(!output.diagnostics.recorded_carry);
            assert(output.diagnostics.inactive_arm_targets_locomotion);
        }
    }
    assert(output.diagnostics.state == RuntimeState::Carry);
    assert(!output.diagnostics.recorded_carry);
    assert(output.diagnostics.inactive_arm_targets_locomotion);
    assert(output.diagnostics.inactive_arm_tracks_locomotion);

    const RuntimeDiagnostics before_failure = runtime.diagnostics();
    RuntimeInput invalid = idle_input(carry_locomotion);
    invalid.locomotion.pose.positions[g1_skeleton::Simulation].x =
        std::numeric_limits<float>::quiet_NaN();
    bool threw = false;
    try {
        (void)runtime.update(invalid);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
    assert(exact(runtime.diagnostics(), before_failure));

    output = advance(runtime, carry_locomotion);
    assert(output.diagnostics.inactive_arm_tracks_locomotion);
    output = runtime.update(reset_input(carry_locomotion));
    assert(output.diagnostics.state == RuntimeState::Locomotion);
    assert(!output.diagnostics.inactive_arm_targets_locomotion);
    assert(!output.diagnostics.inactive_arm_tracks_locomotion);
}

void test_recorded_and_weighted_carry_never_publish_inactive_arm_authority() {
    using namespace interaction;
    for (bool weighted_layered : {false, true}) {
        RuntimeFixture fixture = make_runtime_fixture();
        RuntimeConfig config{};
        if (weighted_layered) {
            config.carry.minimum_root_displacement_m = 10.0F;
            config.carry.inactive_arm_weight = 0.25F;
        }
        InteractionRuntime runtime(
            fixture.database, fixture.features, fixture.registry, config);
        RuntimeOutput output = enter_carry_before_first_update(runtime, fixture);
        assert(!output.diagnostics.inactive_arm_targets_locomotion);
        LocomotionSnapshot carry_locomotion = fixture.locomotion;
        carry_locomotion.pose = output.pose;
        bool saw_recorded = false;
        for (int update = 0; update < 24; ++update) {
            output = advance(runtime, carry_locomotion);
            assert(output.diagnostics.state == RuntimeState::Carry);
            assert(!output.diagnostics.inactive_arm_tracks_locomotion);
            assert(
                output.diagnostics.inactive_arm_targets_locomotion ==
                (!weighted_layered && !output.diagnostics.recorded_carry));
            saw_recorded = saw_recorded || output.diagnostics.recorded_carry;
        }
        if (!weighted_layered) assert(saw_recorded);
    }
}

void test_hand_constraint_weight_tracks_authored_reach_and_attachment() {
    using namespace interaction;
    using namespace interaction::runtime_fixture_detail;
    RuntimeFixture fixture = make_runtime_fixture();
    InteractionRuntime runtime(
        fixture.database, fixture.features, fixture.registry, RuntimeConfig{});

    RuntimeOutput output = runtime.update(interact_input(
        fixture.locomotion, fixture.request));
    assert(output.diagnostics.state == RuntimeState::Preflight);
    assert_valid_hand_constraint_weight(output.diagnostics);
    assert(output.diagnostics.hand_constraint_weight == 0.0F);

    output = advance(runtime, fixture.locomotion);
    assert(output.diagnostics.state == RuntimeState::Align);
    constexpr int32_t kEntryFrame = kFramesPerClip + 10;
    constexpr int32_t kContactFrame =
        kFramesPerClip + kContactLocalFrame;
    assert(output.diagnostics.frame == kEntryFrame);
    assert(output.diagnostics.hand_constraint_weight == 0.0F);

    float previous_weight = output.diagnostics.hand_constraint_weight;
    bool saw_precontact_pickup_replay = false;
    bool saw_contact = false;
    bool saw_attached_pickup_replay = false;
    bool saw_hold = false;
    bool saw_carry = false;
    for (int update = 0; update < kMaximumUpdates; ++update) {
        output = advance(runtime, fixture.locomotion);
        assert_valid_hand_constraint_weight(output.diagnostics);

        if (output.diagnostics.frame <= kContactFrame &&
            (output.diagnostics.state == RuntimeState::Align ||
             output.diagnostics.state == RuntimeState::PickupReplay)) {
            const float expected = std::clamp(
                static_cast<float>(
                    output.diagnostics.frame - kEntryFrame) /
                    static_cast<float>(kContactFrame - kEntryFrame),
                0.0F,
                1.0F);
            assert(output.diagnostics.hand_constraint_weight == expected);
            assert(output.diagnostics.hand_constraint_weight >=
                   previous_weight);
            previous_weight = output.diagnostics.hand_constraint_weight;
            if (output.diagnostics.state == RuntimeState::PickupReplay &&
                output.diagnostics.frame < kContactFrame) {
                saw_precontact_pickup_replay = true;
            }
            if (output.diagnostics.frame == kContactFrame) {
                saw_contact = true;
                assert(output.diagnostics.phase == Phase::Contact);
                assert(output.diagnostics.hand_constraint_weight == 1.0F);
            }
        }

        if (output.diagnostics.attached) {
            assert(output.diagnostics.hand_constraint_weight == 1.0F);
            if (output.diagnostics.state == RuntimeState::PickupReplay) {
                saw_attached_pickup_replay = true;
            }
        }
        if (output.diagnostics.state == RuntimeState::Hold) {
            saw_hold = true;
            assert(output.diagnostics.hand_constraint_weight == 1.0F);
        }
        if (output.diagnostics.state == RuntimeState::Carry) {
            saw_carry = true;
            assert(output.diagnostics.hand_constraint_weight == 1.0F);
            break;
        }
    }
    assert(saw_precontact_pickup_replay);
    assert(saw_contact);
    assert(saw_attached_pickup_replay);
    assert(saw_hold);
    assert(saw_carry);
}

void test_place_hand_constraint_weight_is_full_until_release() {
    using namespace interaction;
    RuntimeFixture fixture = make_runtime_fixture();
    InteractionRuntime runtime(
        fixture.database,
        fixture.features,
        fixture.registry,
        fixture.surface_registry,
        fixture.place_library,
        RuntimeConfig{});

    RuntimeOutput output = enter_carry_before_first_update(runtime, fixture);
    assert(output.diagnostics.hand_constraint_weight == 1.0F);
    const PlaceRequest request = place_request_for(runtime, fixture, 9191U);
    output = runtime.update(place_interact_input(
        fixture.locomotion, request));

    bool saw_preflight = false;
    bool saw_align = false;
    bool saw_replay = false;
    bool saw_release = false;
    for (int update = 0; update < kMaximumUpdates; ++update) {
        const RuntimeState state = output.diagnostics.state;
        if (state == RuntimeState::PlacePreflight ||
            state == RuntimeState::PlaceAlign ||
            state == RuntimeState::PlaceReplay) {
            assert(output.diagnostics.attached);
            assert(output.diagnostics.object_state == ObjectState::Held);
            assert(output.owns_pose);
            assert(output.diagnostics.hand_constraint_weight == 1.0F);
            saw_preflight = saw_preflight ||
                state == RuntimeState::PlacePreflight;
            saw_align = saw_align || state == RuntimeState::PlaceAlign;
            saw_replay = saw_replay || state == RuntimeState::PlaceReplay;
        } else if (state == RuntimeState::PlaceRelease) {
            assert(!output.diagnostics.attached);
            assert(output.diagnostics.object_state == ObjectState::Free);
            assert(output.diagnostics.hand_constraint_weight == 0.0F);
            saw_release = true;
        } else if (state == RuntimeState::Locomotion) {
            assert(output.diagnostics.hand_constraint_weight == 0.0F);
            break;
        } else {
            assert(false && "unexpected state during placement weight proof");
        }
        output = advance(runtime, fixture.locomotion);
    }
    assert(output.diagnostics.state == RuntimeState::Locomotion);
    assert(saw_preflight && saw_align && saw_replay && saw_release);
}

void test_hand_constraint_weight_resets_on_cancel_failure_and_reset() {
    using namespace interaction;

    {
        RuntimeFixture fixture = make_runtime_fixture();
        InteractionRuntime runtime(
            fixture.database,
            fixture.features,
            fixture.registry,
            RuntimeConfig{});
        RuntimeOutput output = enter_align(runtime, fixture);
        while (output.diagnostics.hand_constraint_weight == 0.0F) {
            output = advance(runtime, fixture.locomotion);
            assert(output.diagnostics.state == RuntimeState::Align);
        }
        output = runtime.update(cancel_input(fixture.locomotion));
        assert(output.diagnostics.state == RuntimeState::Locomotion);
        assert(output.diagnostics.result == ResultCode::Cancelled);
        assert(output.diagnostics.hand_constraint_weight == 0.0F);
    }

    {
        RuntimeFixture fixture = contact_failure_fixture();
        RuntimeConfig config{};
        config.ik.accepted_position_m =
            config.ik.maximum_request_position_m;
        InteractionRuntime runtime(
            fixture.database,
            fixture.features,
            fixture.registry,
            config);
        RuntimeOutput output = runtime.update(interact_input(
            fixture.locomotion, fixture.request));
        assert(output.diagnostics.state == RuntimeState::Preflight);
        output = advance(runtime, fixture.locomotion);
        assert(output.diagnostics.state == RuntimeState::Locomotion);
        assert(output.diagnostics.result == ResultCode::Rejected);
        assert(output.diagnostics.reason == Reason::CorrectionLimit);
        assert_valid_hand_constraint_weight(output.diagnostics);
        assert(output.diagnostics.hand_constraint_weight == 0.0F);
    }

    {
        RuntimeFixture fixture = make_runtime_fixture();
        InteractionRuntime runtime(
            fixture.database,
            fixture.features,
            fixture.registry,
            RuntimeConfig{});
        RuntimeOutput output = enter_align(runtime, fixture);
        output = advance_until(
            runtime,
            fixture.locomotion,
            RuntimeState::PickupReplay,
            RuntimeState::Align);
        output = advance_until(
            runtime,
            fixture.locomotion,
            RuntimeState::Hold,
            RuntimeState::PickupReplay);
        output = advance_until(
            runtime,
            fixture.locomotion,
            RuntimeState::Carry,
            RuntimeState::Hold);
        assert(output.diagnostics.hand_constraint_weight == 1.0F);
        output = runtime.update(reset_input(fixture.locomotion));
        assert(output.diagnostics.state == RuntimeState::Locomotion);
        assert(output.diagnostics.result == ResultCode::Reset);
        assert(output.diagnostics.hand_constraint_weight == 0.0F);
        output = advance(runtime, fixture.locomotion);
        assert(output.diagnostics.hand_constraint_weight == 0.0F);
    }
}

void test_carry_reset_preserves_a_newer_authoritative_generation() {
    using namespace interaction;
    RuntimeFixture fixture = make_runtime_fixture();
    InteractionRuntime runtime(
        fixture.database,
        fixture.features,
        fixture.registry,
        RuntimeConfig{});
    RuntimeOutput output = enter_align(runtime, fixture);
    output = advance_until(
        runtime,
        fixture.locomotion,
        RuntimeState::PickupReplay,
        RuntimeState::Align);
    output = advance_until(
        runtime,
        fixture.locomotion,
        RuntimeState::Hold,
        RuntimeState::PickupReplay);
    output = advance_until(
        runtime,
        fixture.locomotion,
        RuntimeState::Carry,
        RuntimeState::Hold);
    assert(output.diagnostics.attached);

    Transform authoritative = fixture.original_object_world;
    authoritative.position.x += 0.45F;
    authoritative.position.z -= 0.20F;
    const TargetHandle newer = fixture.registry.replace_pose(
        fixture.request.target.id, authoritative);
    assert(newer.generation == fixture.request.target.generation + 1U);

    output = runtime.update(reset_input(fixture.locomotion));
    assert(output.diagnostics.state == RuntimeState::Locomotion);
    assert(output.diagnostics.result == ResultCode::Failed);
    assert(output.diagnostics.reason == Reason::TargetChanged);
    assert(!output.diagnostics.attached);
    assert(!output.diagnostics.inactive_arm_tracks_locomotion);
    assert(!output.owns_pose && !output.suppress_steering);
    assert(exact(output.object_world, authoritative));

    const InteractionTarget* preserved = fixture.registry.find(newer);
    assert(preserved != nullptr);
    assert(preserved->state == ObjectState::Free);
    assert(preserved->owner_request == 0U);
    assert(exact(preserved->object_world, authoritative));

    const RuntimeOutput idle = advance(runtime, fixture.locomotion);
    assert(idle.diagnostics.result == ResultCode::Failed);
    assert(idle.diagnostics.reason == Reason::TargetChanged);
    assert(exact(idle.object_world, authoritative));
    assert(fixture.registry.find(newer) != nullptr);
}

void test_carry_update_preserves_a_newer_authoritative_generation() {
    using namespace interaction;
    RuntimeFixture fixture = make_runtime_fixture();
    InteractionRuntime runtime(
        fixture.database,
        fixture.features,
        fixture.registry,
        RuntimeConfig{});
    RuntimeOutput output = enter_align(runtime, fixture);
    output = advance_until(
        runtime,
        fixture.locomotion,
        RuntimeState::PickupReplay,
        RuntimeState::Align);
    output = advance_until(
        runtime,
        fixture.locomotion,
        RuntimeState::Hold,
        RuntimeState::PickupReplay);
    output = advance_until(
        runtime,
        fixture.locomotion,
        RuntimeState::Carry,
        RuntimeState::Hold);
    assert(output.diagnostics.attached);
    const InteractionTarget* held = fixture.registry.find(
        fixture.request.target);
    assert(held != nullptr);
    assert(held->state == ObjectState::Held);
    assert(held->owner_request == fixture.request.request_id);

    Transform authoritative = fixture.original_object_world;
    authoritative.position.x -= 0.35F;
    authoritative.position.z += 0.25F;
    const TargetHandle newer = fixture.registry.replace_pose(
        fixture.request.target.id, authoritative);
    assert(newer.generation == fixture.request.target.generation + 1U);
    LocomotionSnapshot current = fixture.locomotion;
    current.pose.positions[g1_skeleton::Hips].x += 0.17F;

    output = advance(runtime, current);

    assert(output.diagnostics.state == RuntimeState::Locomotion);
    assert(output.diagnostics.result == ResultCode::Failed);
    assert(output.diagnostics.reason == Reason::TargetChanged);
    assert(output.diagnostics.target == newer);
    assert(output.diagnostics.object_state == ObjectState::Free);
    assert(!output.diagnostics.attached);
    assert(!output.diagnostics.recorded_carry);
    assert(!output.diagnostics.inactive_arm_tracks_locomotion);
    assert(!output.owns_pose && !output.suppress_steering);
    assert(exact(output.pose, current.pose));
    assert(exact(output.object_world, authoritative));

    const InteractionTarget* preserved = fixture.registry.find(newer);
    assert(preserved != nullptr);
    assert(preserved->state == ObjectState::Free);
    assert(preserved->owner_request == 0U);
    assert(exact(preserved->object_world, authoritative));

    const RuntimeOutput idle = advance(runtime, current);
    assert(idle.diagnostics.target == newer);
    assert(exact(idle.object_world, authoritative));
    assert(fixture.registry.find(newer) != nullptr);
}

void test_align_cancel_releases_reservation() {
    using namespace interaction;
    RuntimeFixture fixture = make_runtime_fixture();
    InteractionRuntime runtime(
        fixture.database, fixture.features, fixture.registry, RuntimeConfig{});

    RuntimeOutput output = runtime.update(interact_input(
        fixture.locomotion, fixture.request));
    assert(output.diagnostics.state == RuntimeState::Preflight);
    output = advance(runtime, fixture.locomotion);
    assert(output.diagnostics.state == RuntimeState::Align);
    assert(fixture.registry.validate(
        fixture.request.target, fixture.request.request_id));

    output = runtime.update(cancel_input(fixture.locomotion));
    assert(output.diagnostics.state == RuntimeState::Locomotion);
    assert(output.diagnostics.result == ResultCode::Cancelled);
    assert(output.diagnostics.reason == Reason::Cancelled);
    assert(!output.diagnostics.attached);
    assert(!output.owns_pose && !output.suppress_steering);
    assert_free(fixture.registry, fixture.request.target);
}

void test_entry_blends_the_whole_pose_for_quarter_second() {
    using namespace interaction;
    RuntimeFixture fixture = make_runtime_fixture();
    fixture.locomotion.pose.positions[g1_skeleton::Hips].x += 0.30F;
    const float locomotion_hips_x =
        fixture.locomotion.pose.positions[g1_skeleton::Hips].x;
    InteractionRuntime runtime(
        fixture.database, fixture.features, fixture.registry, RuntimeConfig{});

    RuntimeOutput output = runtime.update(interact_input(
        fixture.locomotion, fixture.request));
    assert(exact(output.pose, fixture.locomotion.pose));
    output = advance(runtime, fixture.locomotion);
    assert(output.diagnostics.state == RuntimeState::Align);
    assert(exact(output.pose, fixture.locomotion.pose));

    for (int update = 0; update < 3; ++update) {
        output = advance(runtime, fixture.locomotion);
    }
    const float midpoint_x =
        output.pose.positions[g1_skeleton::Hips].x;
    assert(midpoint_x < locomotion_hips_x);
    assert(midpoint_x > 0.0F);

    for (int update = 3; update < 7; ++update) {
        output = advance(runtime, fixture.locomotion);
    }
    assert(std::abs(
        output.pose.positions[g1_skeleton::Hips].x) < 1.0e-6F);
}

void test_entry_blend_continues_after_early_commit() {
    using namespace interaction;
    RuntimeFixture early_fixture = make_runtime_fixture();
    RuntimeFixture align_fixture = make_runtime_fixture();
    early_fixture.locomotion.pose.positions[g1_skeleton::Hips].x += 0.30F;
    align_fixture.locomotion = early_fixture.locomotion;

    RuntimeConfig early_config{};
    early_config.playback.entry_blend_seconds = 0.50F;
    early_config.playback.commit_horizon_seconds = kDt;
    RuntimeConfig align_config = early_config;
    align_config.playback.commit_horizon_seconds = 0.50F;

    InteractionRuntime early(
        early_fixture.database,
        early_fixture.features,
        early_fixture.registry,
        early_config);
    InteractionRuntime align(
        align_fixture.database,
        align_fixture.features,
        align_fixture.registry,
        align_config);
    RuntimeOutput early_output = enter_align(early, early_fixture);
    RuntimeOutput align_output = enter_align(align, align_fixture);

    early_output = advance(early, early_fixture.locomotion);
    align_output = advance(align, align_fixture.locomotion);
    assert(early_output.diagnostics.state == RuntimeState::PickupReplay);
    assert(align_output.diagnostics.state == RuntimeState::Align);
    assert(exact(early_output.pose, align_output.pose));

    early_output = advance(early, early_fixture.locomotion);
    align_output = advance(align, align_fixture.locomotion);
    assert(early_output.diagnostics.state == RuntimeState::PickupReplay);
    assert(align_output.diagnostics.state == RuntimeState::Align);
    assert(exact(early_output.pose, align_output.pose));
}

void test_cancel_after_commit_before_contact_releases_pickup() {
    using namespace interaction;
    RuntimeFixture fixture = make_runtime_fixture();
    const Transform original =
        fixture.registry.find(fixture.request.target)->object_world;
    InteractionRuntime runtime(
        fixture.database, fixture.features, fixture.registry, RuntimeConfig{});

    RuntimeOutput output = runtime.update(interact_input(
        fixture.locomotion, fixture.request));
    output = advance(runtime, fixture.locomotion);
    while (output.diagnostics.state == RuntimeState::Align) {
        output = advance(runtime, fixture.locomotion);
    }
    assert(output.diagnostics.state == RuntimeState::PickupReplay);
    assert(!output.diagnostics.attached);

    output = runtime.update(cancel_input(fixture.locomotion));
    assert(output.diagnostics.state == RuntimeState::Locomotion);
    assert(output.diagnostics.result == ResultCode::Cancelled);
    assert(output.diagnostics.reason == Reason::Cancelled);
    assert(output.diagnostics.object_state == ObjectState::Free);
    assert(!output.diagnostics.attached);
    assert(!output.owns_pose && !output.suppress_steering);
    assert_free(fixture.registry, fixture.request.target);
    assert(exact(
        fixture.registry.find(fixture.request.target)->object_world,
        original));
}

void test_cancel_after_contact_preserves_attached_pickup() {
    using namespace interaction;
    RuntimeFixture fixture = make_runtime_fixture();
    InteractionRuntime runtime(
        fixture.database, fixture.features, fixture.registry, RuntimeConfig{});

    RuntimeOutput output = runtime.update(interact_input(
        fixture.locomotion, fixture.request));
    output = advance(runtime, fixture.locomotion);
    for (int update = 0;
         update < kMaximumUpdates && !output.diagnostics.attached;
         ++update) {
        output = advance(runtime, fixture.locomotion);
    }
    assert(output.diagnostics.attached);
    assert(output.diagnostics.state == RuntimeState::PickupReplay ||
           output.diagnostics.state == RuntimeState::Hold);
    const int32_t attached_frame = output.diagnostics.frame;

    output = runtime.update(cancel_input(fixture.locomotion));
    assert(output.diagnostics.state == RuntimeState::PickupReplay ||
           output.diagnostics.state == RuntimeState::Hold ||
           output.diagnostics.state == RuntimeState::Carry);
    assert(output.diagnostics.attached);
    assert(output.diagnostics.frame >= attached_frame);
    assert(fixture.registry.validate(
        fixture.request.target, fixture.request.request_id));
}

void test_commit_at_contact_preserves_the_one_shot_crossing() {
    using namespace interaction;
    RuntimeFixture fixture = make_runtime_fixture();
    RuntimeConfig config{};
    constexpr int32_t kEntryFrame =
        runtime_fixture_detail::kFramesPerClip + 10;
    config.playback.commit_horizon_seconds =
        fixture.database.time_to_contact.at(kEntryFrame);
    InteractionRuntime runtime(
        fixture.database, fixture.features, fixture.registry, config);

    RuntimeOutput output = runtime.update(interact_input(
        fixture.locomotion, fixture.request));
    assert(output.diagnostics.state == RuntimeState::Preflight);
    output = advance(runtime, fixture.locomotion);
    assert(output.diagnostics.state == RuntimeState::Align);
    output = advance_until(
        runtime,
        fixture.locomotion,
        RuntimeState::PickupReplay,
        RuntimeState::Align);
    assert(output.diagnostics.frame >=
        runtime_fixture_detail::kFramesPerClip +
        runtime_fixture_detail::kContactLocalFrame);

    output = advance_until(
        runtime,
        fixture.locomotion,
        RuntimeState::Hold,
        RuntimeState::PickupReplay);
    assert(output.diagnostics.attached);
    output = advance_until(
        runtime,
        fixture.locomotion,
        RuntimeState::Carry,
        RuntimeState::Hold);
    assert(output.diagnostics.result == ResultCode::Succeeded);
}

void test_nonunit_playback_speed_commits_by_source_contact() {
    using namespace interaction;
    RuntimeFixture fixture = make_runtime_fixture();
    RuntimeConfig config{};
    config.playback.speed = 1.10F;
    config.playback.commit_horizon_seconds = 1.0F;
    config.playback.maximum_alignment_seconds = 1.0F;
    InteractionRuntime runtime(
        fixture.database, fixture.features, fixture.registry, config);
    RuntimeOutput output = enter_align(runtime, fixture);

    constexpr int32_t kContactFrame =
        runtime_fixture_detail::kFramesPerClip +
        runtime_fixture_detail::kContactLocalFrame;
    for (int update = 0; update < 13; ++update) {
        output = advance(runtime, fixture.locomotion);
        assert(output.diagnostics.state == RuntimeState::Align);
        assert(output.diagnostics.frame < kContactFrame);
    }

    output = advance(runtime, fixture.locomotion);
    assert(output.diagnostics.state == RuntimeState::PickupReplay);
    assert(output.diagnostics.frame == kContactFrame);
    assert(output.diagnostics.playback_speed == config.playback.speed);
}

void test_realized_transition_rejects_contact_that_cannot_attach() {
    using namespace interaction;
    RuntimeFixture fixture = contact_failure_fixture();
    RuntimeConfig config{};
    config.ik.accepted_position_m =
        config.ik.maximum_request_position_m;
    const MatchInput matcher_input = match_input_for(fixture);
    const MatchResult selected = select_whole_clip(
        matcher_input, config.matcher);
    assert(selected.accepted);

    const runtime_detail::RealizedPickTransitionEvaluation realized =
        runtime_detail::evaluate_realized_pick_transition(
            fixture.database,
            fixture.locomotion.pose,
            selected.candidate,
            matcher_input.target,
            matcher_input.affordance,
            config.playback,
            config.ik);
    assert(!realized.feasible);
    assert(realized.reason == Reason::ContactPosition);
    assert(realized.frame == selected.candidate.contact_frame);
}

void test_realized_transition_rejects_invalid_reconstructed_object() {
    using namespace interaction;
    RuntimeFixture fixture = make_runtime_fixture();
    const RuntimeConfig config{};
    MatchInput matcher_input = match_input_for(fixture);
    const MatchResult selected = select_whole_clip(
        matcher_input, config.matcher);
    assert(selected.accepted);

    MatchCandidate candidate = selected.candidate;
    const quat quarter_turn = quat_from_angle_axis(
        0.25F * 3.14159265358979323846F,
        vec3(0.0F, 0.0F, 1.0F));
    const WorldPose source_contact_pose = world_pose(pose_at_frame(
        fixture.database, candidate.contact_frame));
    const Transform source_contact_hand = {
        source_contact_pose.positions[kRightHandBone],
        source_contact_pose.rotations[kRightHandBone],
    };
    const vec3 desired_contact(0.0F, 0.0F, 3.0F);
    candidate.scene_from_source.rotation = quarter_turn;
    candidate.scene_from_source.position = desired_contact -
        quat_mul_vec3(
            quarter_turn, source_contact_hand.position);

    Pose live_entry_pose = fixture.locomotion.pose;
    const Transform live_root = compose(
        candidate.scene_from_source,
        Transform{
            live_entry_pose.positions[g1_skeleton::Simulation],
            live_entry_pose.rotations[g1_skeleton::Simulation],
        });
    live_entry_pose.positions[g1_skeleton::Simulation] =
        live_root.position;
    live_entry_pose.rotations[g1_skeleton::Simulation] =
        live_root.rotation;

    InteractionTarget target = matcher_input.target;
    target.table_world = compose(
        candidate.scene_from_source, target.table_world);
    const float maximum = std::numeric_limits<float>::max();
    target.object_world = {
        vec3(-maximum, -maximum, desired_contact.z), quat()};
    GraspAffordance affordance = matcher_input.affordance;
    affordance.hand_in_object = {
        vec3(maximum, maximum, 0.0F), quarter_turn};
    for (GraspAffordance& authored : target.affordances) {
        if (authored.id == affordance.id) authored = affordance;
    }
    const Transform target_hand = compose(
        target.object_world, affordance.hand_in_object);
    assert(std::isfinite(target_hand.position.x));
    assert(std::isfinite(target_hand.position.y));
    assert(std::isfinite(target_hand.position.z));
    assert(target_hand.position.x == desired_contact.x);
    assert(target_hand.position.y == desired_contact.y);
    assert(target_hand.position.z == desired_contact.z);
    const Transform mapped_source_contact = compose(
        candidate.scene_from_source, source_contact_hand);
    assert(length(mapped_source_contact.position - target_hand.position) ==
           0.0F);
    assert(quat_angle_between(
               mapped_source_contact.rotation,
               target_hand.rotation) == 0.0F);
    const Transform reconstructed_object = compose(
        target_hand, inverse(affordance.hand_in_object));
    assert(!std::isfinite(reconstructed_object.position.x) ||
           !std::isfinite(reconstructed_object.position.y) ||
           !std::isfinite(reconstructed_object.position.z));

    const runtime_detail::RealizedPickTransitionEvaluation realized =
        runtime_detail::evaluate_realized_pick_transition(
            fixture.database,
            live_entry_pose,
            candidate,
            target,
            affordance,
            config.playback,
            config.ik,
            config.attachment);
    assert(!realized.feasible);
    assert(realized.reason == Reason::ContactOrientation);
    assert(realized.frame == candidate.contact_frame);
}

void test_realized_contact_reason_adapter_preserves_matcher_categories() {
    using namespace interaction;
    assert(runtime_detail::realized_pick_hard_rejection_reason(
               Reason::ContactPosition) == Reason::CorrectionLimit);
    assert(runtime_detail::realized_pick_hard_rejection_reason(
               Reason::ContactOrientation) == Reason::CorrectionLimit);
    assert(runtime_detail::realized_pick_hard_rejection_reason(
               Reason::JointLimit) == Reason::CorrectionLimit);
    assert(runtime_detail::realized_pick_hard_rejection_reason(
               Reason::LostContact) == Reason::NoCandidate);
    assert(runtime_detail::realized_pick_hard_rejection_reason(
               Reason::BlockedPath) == Reason::BlockedPath);
    assert(runtime_detail::realized_pick_hard_rejection_reason(
               Reason::None) == Reason::None);
}

void test_canonical_updates_reject_invalid_contact_before_commit() {
    using namespace interaction;
    RuntimeFixture fixture = contact_failure_fixture();
    RuntimeConfig config{};
    config.ik.accepted_position_m =
        config.ik.maximum_request_position_m;
    InteractionRuntime runtime(
        fixture.database, fixture.features, fixture.registry, config);

    const PickEntryPreview preview = runtime.preview_pick(
        fixture.locomotion,
        live_pick_entry_root(fixture.locomotion),
        fixture.request.target,
        fixture.request.affordance_id);
    assert_preview_rejected(preview, Reason::CorrectionLimit);

    RuntimeOutput output = runtime.update(interact_input(
        fixture.locomotion, fixture.request));
    assert(output.diagnostics.state == RuntimeState::Preflight);
    output = advance(runtime, fixture.locomotion);
    assert(output.diagnostics.state == RuntimeState::Locomotion);
    assert(output.diagnostics.result == ResultCode::Rejected);
    assert(output.diagnostics.reason == Reason::CorrectionLimit);
    assert(!output.diagnostics.attached);
}

void test_canonical_updates_cannot_skip_post_attach_contact_loss() {
    using namespace interaction;
    RuntimeFixture fixture = post_attach_contact_loss_fixture();
    InteractionRuntime runtime(
        fixture.database,
        fixture.features,
        fixture.registry,
        RuntimeConfig{});
    RuntimeOutput output = enter_align(runtime, fixture);

    for (int update = 0;
         update < kMaximumUpdates &&
         output.diagnostics.result != ResultCode::Failed &&
         output.diagnostics.state != RuntimeState::Carry;
         ++update) {
        output = advance(runtime, fixture.locomotion);
    }
    assert(output.diagnostics.result == ResultCode::Failed);
    assert(output.diagnostics.reason == Reason::LostContact);
    assert(!output.diagnostics.attached);
}

void test_pickup_success_precedes_later_canonical_contact_loss() {
    using namespace interaction;
    RuntimeFixture fixture = post_success_contact_loss_fixture();
    RuntimeConfig config{};
    config.attachment.required_lift_m = 0.0F;
    config.attachment.required_hold_seconds = 0.50F;
    InteractionRuntime runtime(
        fixture.database,
        fixture.features,
        fixture.registry,
        config);
    RuntimeOutput output = enter_align(runtime, fixture);

    for (int update = 0;
         update < kMaximumUpdates &&
         output.diagnostics.state != RuntimeState::Carry &&
         output.diagnostics.result != ResultCode::Failed;
         ++update) {
        output = advance(runtime, fixture.locomotion);
    }

    constexpr int32_t kFirstHoldFrame =
        runtime_fixture_detail::kFramesPerClip +
        runtime_fixture_detail::kHoldLocalFrame;
    assert(output.diagnostics.state == RuntimeState::Carry);
    assert(output.diagnostics.result == ResultCode::Succeeded);
    assert(output.diagnostics.frame == kFirstHoldFrame);
    assert(output.diagnostics.object_state == ObjectState::Held);
}

void test_hold_interval_precedes_its_canonical_stop_frame_event() {
    using namespace interaction;
    RuntimeFixture fixture = post_success_contact_loss_fixture(0);
    RuntimeConfig config{};
    config.attachment.required_lift_m = 0.0F;
    config.attachment.required_hold_seconds = 0.58F;
    InteractionRuntime runtime(
        fixture.database, fixture.features, fixture.registry, config);
    RuntimeOutput output = enter_align(runtime, fixture);
    for (int update = 0;
         update < kMaximumUpdates &&
         output.diagnostics.state != RuntimeState::Carry &&
         output.diagnostics.result != ResultCode::Failed;
         ++update) {
        output = advance(runtime, fixture.locomotion);
    }
    assert(output.diagnostics.state == RuntimeState::Carry);
    assert(output.diagnostics.result == ResultCode::Succeeded);
    assert(output.diagnostics.frame ==
        runtime_fixture_detail::kFramesPerClip +
            runtime_fixture_detail::kHoldLocalFrame);
}

void test_blocked_reach_transition_falls_back_to_safe_approach_entry() {
    using namespace interaction;
    RuntimeFixture fixture = entry_blend_table_crossing_fixture();
    RuntimeConfig config{};
    config.ik.accepted_position_m = 0.01F;
    config.ik.maximum_iterations = 64;
    config.attachment.maximum_position_error_m = 0.02F;
    InteractionRuntime runtime(
        fixture.database, fixture.features, fixture.registry, config);

    constexpr int32_t kExpectedApproachEntry =
        runtime_fixture_detail::kFramesPerClip;
    constexpr int32_t kExpectedReachEntry =
        runtime_fixture_detail::kFramesPerClip + 10;
    constexpr int32_t kExpectedContact =
        runtime_fixture_detail::kFramesPerClip +
        runtime_fixture_detail::kContactLocalFrame;

    const MatchInput matcher_input = match_input_for(fixture);
    const vec3 live_hand = world_pose(fixture.locomotion.pose).positions[
        kRightHandBone];
    const float expanded_table_top =
        matcher_input.target.table_world.position.y +
        0.5F * matcher_input.target.table_size.y +
        matcher_input.affordance.clearance_radius;
    const float expanded_table_front =
        matcher_input.target.table_world.position.z -
        0.5F * matcher_input.target.table_size.z -
        matcher_input.affordance.clearance_radius;
    assert(live_hand.y < expanded_table_top);
    assert(live_hand.z < expanded_table_front);
    const vec3 authored_approach_hand = world_pose(pose_at_frame(
        fixture.database, kExpectedApproachEntry)).positions[kRightHandBone];
    const vec3 authored_reach_hand = world_pose(pose_at_frame(
        fixture.database, kExpectedReachEntry)).positions[kRightHandBone];
    assert(authored_approach_hand.y > expanded_table_top);
    assert(authored_approach_hand.z < expanded_table_front);
    assert(authored_reach_hand.y > expanded_table_top);
    assert(authored_reach_hand.z < expanded_table_front);
    const MatchResult unfiltered = select_whole_clip(
        matcher_input, config.matcher);
    assert(unfiltered.accepted);
    assert(unfiltered.candidate.entry_frame == kExpectedReachEntry);
    const runtime_detail::RealizedPickTransitionEvaluation blocked_reach =
        runtime_detail::evaluate_realized_pick_transition(
            fixture.database,
            fixture.locomotion.pose,
            unfiltered.candidate,
            matcher_input.target,
            matcher_input.affordance,
            config.playback,
            config.ik);
    assert(!blocked_reach.feasible);
    assert(blocked_reach.reason == Reason::BlockedPath);

    const matcher_detail::PickEvaluation safe_match =
        matcher_detail::evaluate_pick_entries(
            {
                matcher_input.database,
                matcher_input.features,
                matcher_input.query,
                matcher_input.locomotion,
                matcher_input.target,
                matcher_input.affordance,
            },
            config.matcher,
            [&](const MatchCandidate& candidate) {
                const runtime_detail::RealizedPickTransitionEvaluation
                    realized = runtime_detail::evaluate_realized_pick_transition(
                    fixture.database,
                    fixture.locomotion.pose,
                    candidate,
                    matcher_input.target,
                    matcher_input.affordance,
                    config.playback,
                    config.ik);
                return realized.reason;
            });
    assert(safe_match.selection.accepted);
    assert(safe_match.selection.candidate.entry_frame ==
           kExpectedApproachEntry);
    const runtime_detail::RealizedPickTransitionEvaluation safe_approach =
        runtime_detail::evaluate_realized_pick_transition(
            fixture.database,
            fixture.locomotion.pose,
            safe_match.selection.candidate,
            matcher_input.target,
            matcher_input.affordance,
            config.playback,
            config.ik);
    assert(safe_approach.feasible);
    assert(safe_approach.reason == Reason::None);

    const PickEntryPreview preview = runtime.preview_pick(
        fixture.locomotion,
        live_pick_entry_root(fixture.locomotion),
        fixture.request.target,
        fixture.request.affordance_id);
    assert(preview.path_feasible);
    assert(preview.match_ready);
    assert(preview.path_reason == Reason::None);
    assert(preview.match_reason == Reason::None);
    assert(preview.match_candidate.entry_frame == kExpectedApproachEntry);
    assert(preview.match_candidate.contact_frame == kExpectedContact);

    RuntimeOutput output = runtime.update(interact_input(
        fixture.locomotion, fixture.request));
    assert(output.diagnostics.state == RuntimeState::Preflight);
    output = advance(runtime, fixture.locomotion);
    assert(output.diagnostics.state == RuntimeState::Align);
    assert(output.diagnostics.frame == kExpectedApproachEntry);

    for (int update = 0;
         update < kMaximumUpdates &&
         output.diagnostics.state != RuntimeState::Carry &&
         output.diagnostics.result != ResultCode::Failed;
         ++update) {
        output = advance(runtime, fixture.locomotion);
    }
    assert(output.diagnostics.state == RuntimeState::Carry);
    assert(output.diagnostics.result == ResultCode::Succeeded);
    assert(output.diagnostics.reason == Reason::None);
    assert(output.diagnostics.attached);
    assert_held_by_original_owner(fixture);
}

void test_nonunit_fractional_wall_endpoint_is_certified_before_commit() {
    using namespace interaction;
    RuntimeConfig config{};
    config.playback.speed = 0.85F;

    RuntimeFixture trace_fixture = make_nonunit_fractional_arc_fixture();
    InteractionRuntime trace_runtime(
        trace_fixture.database,
        trace_fixture.features,
        trace_fixture.registry,
        config);
    RuntimeOutput trace = enter_align(trace_runtime, trace_fixture);
    float previous_clearance_elapsed =
        InteractionRuntimeTestAccess::clearance_elapsed_seconds(
            trace_runtime);
    assert(previous_clearance_elapsed == 0.0F);
    assert(InteractionRuntimeTestAccess::player_elapsed_seconds(
               trace_runtime) == previous_clearance_elapsed);
    assert(InteractionRuntimeTestAccess::clearance_frame(trace_runtime) ==
           InteractionRuntimeTestAccess::player_frame(trace_runtime));
    for (int update = 0; update < 20; ++update) {
        if (trace.diagnostics.state != RuntimeState::Align) break;
        trace = advance(trace_runtime, trace_fixture.locomotion);
        const float player_elapsed =
            InteractionRuntimeTestAccess::player_elapsed_seconds(
                trace_runtime);
        const float clearance_elapsed =
            InteractionRuntimeTestAccess::clearance_elapsed_seconds(
                trace_runtime);
        assert(clearance_elapsed > previous_clearance_elapsed);
        assert(near(clearance_elapsed, player_elapsed, 1.0e-6F));
        assert(InteractionRuntimeTestAccess::clearance_frame(trace_runtime) ==
               InteractionRuntimeTestAccess::player_frame(trace_runtime));
        previous_clearance_elapsed = clearance_elapsed;
    }
    assert(trace.diagnostics.state == RuntimeState::PickupReplay);
    trace = advance(trace_runtime, trace_fixture.locomotion);
    assert(trace.diagnostics.state == RuntimeState::PickupReplay);
    constexpr int32_t kFractionalLeftFrame =
        runtime_fixture_detail::kFramesPerClip + 21;
    assert(trace.diagnostics.frame == kFractionalLeftFrame);
    const vec3 fractional_hand = world_pose(trace.pose).positions[
        kRightHandBone];

    RuntimeFixture fixture = make_nonunit_fractional_arc_fixture();
    set_fractional_endpoint_table(fixture, fractional_hand);
    const MatchInput input = match_input_for(fixture);
    const MatchResult authored = select_whole_clip(input, config.matcher);
    assert(authored.accepted);
    assert(authored.candidate.entry_frame ==
           runtime_fixture_detail::kFramesPerClip + 10);
    const vec3 integer_start = world_pose(pose_at_frame(
        fixture.database, kFractionalLeftFrame)).positions[kRightHandBone];
    const vec3 integer_stop = world_pose(pose_at_frame(
        fixture.database, kFractionalLeftFrame + 1)).positions[
            kRightHandBone];
    assert(length(fractional_hand - integer_start) > 0.10F);
    assert(length(fractional_hand - integer_stop) > 0.10F);

    const runtime_detail::RealizedPickTransitionEvaluation reach_evaluation =
        runtime_detail::evaluate_realized_pick_transition(
            fixture.database,
            fixture.locomotion.pose,
            authored.candidate,
            input.target,
            input.affordance,
            config.playback,
            config.ik);
    if (reach_evaluation.feasible) {
        InteractionRuntime mismatch(
            fixture.database,
            fixture.features,
            fixture.registry,
            config);
        RuntimeOutput output = enter_align(mismatch, fixture);
        for (int update = 0;
             update < kMaximumUpdates &&
             output.diagnostics.result != ResultCode::Failed &&
             output.diagnostics.state != RuntimeState::Carry;
             ++update) {
            output = advance(mismatch, fixture.locomotion);
        }
        assert(output.diagnostics.result == ResultCode::Failed);
        assert(output.diagnostics.reason == Reason::BlockedPath);
        assert(!output.diagnostics.attached);
    }
    assert(!reach_evaluation.feasible);
    assert(reach_evaluation.reason == Reason::BlockedPath);

    RuntimeFixture execution_fixture =
        make_nonunit_fractional_arc_fixture();
    set_fractional_endpoint_table(execution_fixture, fractional_hand);
    InteractionRuntime runtime(
        execution_fixture.database,
        execution_fixture.features,
        execution_fixture.registry,
        config);
    const PickEntryPreview preview = runtime.preview_pick(
        execution_fixture.locomotion,
        live_pick_entry_root(execution_fixture.locomotion),
        execution_fixture.request.target,
        execution_fixture.request.affordance_id);
    assert(preview.path_feasible);
    assert(preview.match_ready);
    assert(execution_fixture.database.phases.at(
               static_cast<size_t>(preview.match_candidate.entry_frame)) ==
           static_cast<uint8_t>(Phase::Approach));
    RuntimeOutput output = runtime.update(interact_input(
        execution_fixture.locomotion, execution_fixture.request));
    assert(output.diagnostics.state == RuntimeState::Preflight);
    output = advance(runtime, execution_fixture.locomotion);
    assert(output.diagnostics.state == RuntimeState::Align);
    assert(output.diagnostics.frame ==
           preview.match_candidate.entry_frame);
    for (int update = 0;
         update < kMaximumUpdates &&
         output.diagnostics.state != RuntimeState::Carry &&
         output.diagnostics.result != ResultCode::Failed;
         ++update) {
        output = advance(runtime, execution_fixture.locomotion);
    }
    assert(output.diagnostics.state == RuntimeState::Carry);
    assert(output.diagnostics.result == ResultCode::Succeeded);
    assert(output.diagnostics.reason == Reason::None);
    assert(output.diagnostics.attached);
}

void test_realized_post_ik_table_sweep_is_rejected_in_preflight() {
    using namespace interaction;
    RuntimeFixture fixture = post_ik_table_crossing_fixture();
    const InteractionTarget* authored = fixture.registry.find(
        fixture.request.target);
    const GraspAffordance* affordance = fixture.registry.find_affordance(
        fixture.request.target, fixture.request.affordance_id);
    assert(authored != nullptr && affordance != nullptr);
    const Transform target_hand = compose(
        authored->object_world, affordance->hand_in_object);
    const Transform entry_hand = {
        world_pose(fixture.locomotion.pose).positions[kRightHandBone],
        world_pose(fixture.locomotion.pose).rotations[kRightHandBone],
    };
    const float expanded_half_z =
        0.5F * authored->table_size.z + affordance->clearance_radius;
    assert(entry_hand.position.z <
        authored->table_world.position.z - expanded_half_z);
    assert(target_hand.position.z >
        authored->table_world.position.z + expanded_half_z);

    RuntimeConfig config{};
    config.ik.accepted_position_m = 0.01F;
    config.ik.maximum_iterations = 64;
    config.attachment.maximum_position_error_m = 0.02F;
    InteractionRuntime runtime(
        fixture.database, fixture.features, fixture.registry, config);
    const PickEntryPreview preview = runtime.preview_pick(
        fixture.locomotion,
        live_pick_entry_root(fixture.locomotion),
        fixture.request.target,
        fixture.request.affordance_id);
    assert_preview_rejected(preview, Reason::BlockedPath);
    RuntimeOutput output = runtime.update(interact_input(
        fixture.locomotion, fixture.request));
    assert(output.diagnostics.state == RuntimeState::Preflight);
    output = advance(runtime, fixture.locomotion);
    assert(output.diagnostics.state == RuntimeState::Locomotion);
    assert(output.diagnostics.result == ResultCode::Rejected);
    assert(output.diagnostics.reason == Reason::BlockedPath);
    assert(!output.diagnostics.attached);
}

void test_realized_post_ik_object_sweep_is_rejected_in_preflight() {
    using namespace interaction;
    RuntimeFixture fixture = post_ik_object_crossing_fixture();
    const InteractionTarget* authored = fixture.registry.find(
        fixture.request.target);
    const GraspAffordance* affordance = fixture.registry.find_affordance(
        fixture.request.target, fixture.request.affordance_id);
    assert(authored != nullptr && affordance != nullptr);
    const Transform target_hand = compose(
        authored->object_world, affordance->hand_in_object);
    const float expanded_half_z =
        0.5F * authored->object_dimensions.z +
        affordance->clearance_radius;
    assert(target_hand.position.z >
        authored->object_world.position.z - expanded_half_z);
    assert(target_hand.position.z <
        authored->object_world.position.z + expanded_half_z);

    constexpr int32_t kPreContact =
        runtime_fixture_detail::kFramesPerClip +
        runtime_fixture_detail::kContactLocalFrame - 1;
    const float recorded_precontact_z = world_pose(
        pose_at_frame(fixture.database, kPreContact)).positions[
            kRightHandBone].z;
    const float correction_alpha =
        static_cast<float>(runtime_fixture_detail::kContactLocalFrame - 1 - 10) /
        static_cast<float>(runtime_fixture_detail::kContactLocalFrame - 10);
    const float corrected_precontact_z = recorded_precontact_z +
        correction_alpha * (
            target_hand.position.z - authored->object_world.position.z);
    assert(recorded_precontact_z <
        authored->object_world.position.z - expanded_half_z);
    assert(corrected_precontact_z >
        authored->object_world.position.z - expanded_half_z);
    assert(corrected_precontact_z <
        authored->object_world.position.z + expanded_half_z);

    RuntimeConfig config{};
    config.ik.accepted_position_m = 0.01F;
    config.ik.maximum_iterations = 64;
    config.attachment.maximum_position_error_m = 0.02F;
    InteractionRuntime runtime(
        fixture.database, fixture.features, fixture.registry, config);
    const PickEntryPreview preview = runtime.preview_pick(
        fixture.locomotion,
        live_pick_entry_root(fixture.locomotion),
        fixture.request.target,
        fixture.request.affordance_id);
    assert_preview_rejected(preview, Reason::BlockedPath);
    RuntimeOutput output = runtime.update(interact_input(
        fixture.locomotion, fixture.request));
    assert(output.diagnostics.state == RuntimeState::Preflight);
    output = advance(runtime, fixture.locomotion);
    assert(output.diagnostics.state == RuntimeState::Locomotion);
    assert(output.diagnostics.result == ResultCode::Rejected);
    assert(output.diagnostics.reason == Reason::BlockedPath);
    assert(!output.diagnostics.attached);
}

void test_curved_precontact_blocked_reach_falls_back_to_safe_approach() {
    using namespace interaction;
    RuntimeFixture fixture = curved_precontact_clearance_fixture();
    const InteractionTarget* target = fixture.registry.find(
        fixture.request.target);
    const GraspAffordance* affordance = fixture.registry.find_affordance(
        fixture.request.target, fixture.request.affordance_id);
    assert(target != nullptr && affordance != nullptr);
    const float expanded_half_x =
        0.5F * target->table_size.x + affordance->clearance_radius;
    constexpr int32_t kIntermediateFrame =
        runtime_fixture_detail::kFramesPerClip + 18;
    const float authored_hand_x = world_pose(pose_at_frame(
        fixture.database, kIntermediateFrame)).positions[
            kRightHandBone].x;
    assert(std::abs(authored_hand_x - target->table_world.position.x) >
        expanded_half_x);

    RuntimeConfig config{};
    config.ik.accepted_position_m = 0.01F;
    config.ik.maximum_iterations = 64;
    InteractionRuntime runtime(
        fixture.database, fixture.features, fixture.registry, config);
    const MatchInput matcher_input = match_input_for(fixture);
    const MatchResult unfiltered = select_whole_clip(
        matcher_input, config.matcher);
    assert(unfiltered.accepted);
    assert(fixture.database.phases.at(
               static_cast<size_t>(unfiltered.candidate.entry_frame)) ==
           static_cast<uint8_t>(Phase::Reach));
    const runtime_detail::RealizedPickTransitionEvaluation blocked_reach =
        runtime_detail::evaluate_realized_pick_transition(
            fixture.database,
            fixture.locomotion.pose,
            unfiltered.candidate,
            matcher_input.target,
            matcher_input.affordance,
            config.playback,
            config.ik);
    assert(!blocked_reach.feasible);
    assert(blocked_reach.reason == Reason::BlockedPath);
    const PickEntryPreview preview = runtime.preview_pick(
        fixture.locomotion,
        live_pick_entry_root(fixture.locomotion),
        fixture.request.target,
        fixture.request.affordance_id);
    assert(preview.path_feasible);
    assert(preview.match_ready);
    assert(preview.path_reason == Reason::None);
    assert(preview.match_reason == Reason::None);
    assert(fixture.database.phases.at(
               static_cast<size_t>(
                   preview.match_candidate.entry_frame)) ==
           static_cast<uint8_t>(Phase::Approach));
    RuntimeOutput output = enter_align(runtime, fixture);
    assert(output.diagnostics.frame ==
           preview.match_candidate.entry_frame);
    for (int update = 0;
         update < kMaximumUpdates &&
         output.diagnostics.state != RuntimeState::Carry &&
         output.diagnostics.result != ResultCode::Failed;
         ++update) {
        output = advance(runtime, fixture.locomotion);
    }
    assert(output.diagnostics.state == RuntimeState::Carry);
    assert(output.diagnostics.result == ResultCode::Succeeded);
    assert(output.diagnostics.reason == Reason::None);
    assert(output.diagnostics.attached);
}

void test_runtime_clearance_backstop_rejects_post_certification_change() {
    using namespace interaction;
    RuntimeFixture fixture = make_runtime_fixture();
    InteractionRuntime runtime(
        fixture.database,
        fixture.features,
        fixture.registry,
        RuntimeConfig{});

    const PickEntryPreview preview = runtime.preview_pick(
        fixture.locomotion,
        live_pick_entry_root(fixture.locomotion),
        fixture.request.target,
        fixture.request.affordance_id);
    assert(preview.path_feasible);
    assert(preview.match_ready);
    assert(preview.match_candidate.entry_frame ==
           runtime_fixture_detail::kFramesPerClip + 10);

    RuntimeOutput output = enter_align(runtime, fixture);
    assert(output.diagnostics.frame ==
           preview.match_candidate.entry_frame);
    constexpr int32_t kFuturePrecontactFrame =
        runtime_fixture_detail::kFramesPerClip + 18;
    assert(kFuturePrecontactFrame > output.diagnostics.frame);
    vec3 future_hand = runtime_fixture_detail::read_bone_position(
        fixture.database, kFuturePrecontactFrame, kRightHandBone);
    future_hand.y = 0.60F;
    runtime_fixture_detail::write_bone_position(
        fixture.database,
        kFuturePrecontactFrame,
        kRightHandBone,
        future_hand);

    bool saw_blocked = false;
    for (int update = 0; update < kMaximumUpdates; ++update) {
        output = advance(runtime, fixture.locomotion);
        if (output.diagnostics.result == ResultCode::Failed) {
            assert(output.diagnostics.reason == Reason::BlockedPath);
            assert(!output.diagnostics.attached);
            saw_blocked = true;
            break;
        }
        assert(output.diagnostics.state != RuntimeState::Carry);
    }
    assert(saw_blocked);
}

void test_stale_generation_is_rejected() {
    using namespace interaction;
    RuntimeFixture fixture = make_runtime_fixture();
    const PickRequest stale_request = fixture.request;
    InteractionRuntime runtime(
        fixture.database, fixture.features, fixture.registry, RuntimeConfig{});

    RuntimeOutput output = runtime.update(interact_input(
        fixture.locomotion, stale_request));
    assert(output.diagnostics.state == RuntimeState::Preflight);
    const TargetHandle replacement = fixture.registry.replace_pose(
        stale_request.target.id, fixture.original_object_world);
    assert(replacement.generation == stale_request.target.generation + 1U);

    output = advance(runtime, fixture.locomotion);
    assert(output.diagnostics.state == RuntimeState::Locomotion);
    assert(output.diagnostics.result == ResultCode::Rejected);
    assert(output.diagnostics.reason == Reason::TargetChanged);
    assert(!output.diagnostics.attached);
    assert(!output.owns_pose && !output.suppress_steering);
    assert_free(fixture.registry, replacement);
}

void test_align_rejects_profile_or_bounds_snapshot_changes() {
    using namespace interaction;

    for (int mutation = 0; mutation < 2; ++mutation) {
        RuntimeFixture fixture = make_runtime_fixture();
        InteractionRuntime runtime(
            fixture.database,
            fixture.features,
            fixture.registry,
            RuntimeConfig{});
        RuntimeOutput output = enter_align(runtime, fixture);
        assert(output.diagnostics.state == RuntimeState::Align);

        InteractionTarget* authoritative = fixture.registry.find(
            fixture.request.target);
        assert(authoritative != nullptr);
        if (mutation == 0) {
            ++authoritative->object_profile_id;
        } else {
            authoritative->object_bounds.center_object.z += 0.001F;
        }

        output = advance(runtime, fixture.locomotion);
        assert(output.diagnostics.state == RuntimeState::Locomotion);
        assert(output.diagnostics.result == ResultCode::Rejected);
        assert(output.diagnostics.reason == Reason::TargetChanged);
        assert(!output.diagnostics.attached);
        assert(!output.owns_pose && !output.suppress_steering);
        const InteractionTarget* released = fixture.registry.find(
            fixture.request.target);
        assert(released != nullptr);
        assert(released->state == ObjectState::Free);
        assert(released->owner_request == 0U);
    }
}

void test_failed_reservation_never_releases_an_existing_owner() {
    using namespace interaction;
    RuntimeFixture fixture = make_runtime_fixture();
    assert(fixture.registry.reserve(
        fixture.request.target, fixture.request.request_id));
    InteractionRuntime runtime(
        fixture.database, fixture.features, fixture.registry, RuntimeConfig{});

    RuntimeOutput output = runtime.update(interact_input(
        fixture.locomotion, fixture.request));
    assert(output.diagnostics.state == RuntimeState::Preflight);
    output = advance(runtime, fixture.locomotion);
    assert(output.diagnostics.state == RuntimeState::Locomotion);
    assert(output.diagnostics.result == ResultCode::Rejected);
    assert(output.diagnostics.reason == Reason::TargetChanged);

    const InteractionTarget* target = fixture.registry.find(
        fixture.request.target);
    assert(target != nullptr);
    assert(target->state == ObjectState::Targeted);
    assert(target->owner_request == fixture.request.request_id);
}

void test_runtime_config_is_validated_before_reservation() {
    using namespace interaction;
    RuntimeConfig config{};
    config.matcher.maximum_approach_m = -1.0F;
    assert_invalid_runtime_config(config);

    config = RuntimeConfig{};
    config.matcher.group_weights.fill(0.0F);
    assert_invalid_runtime_config(config);

    config = RuntimeConfig{};
    config.playback.canonical_fps = 30.0F;
    assert_invalid_runtime_config(config);

    config = RuntimeConfig{};
    config.playback.speed = 2.0F;
    assert_invalid_runtime_config(config);

    config = RuntimeConfig{};
    config.ik.damping = 0.0F;
    assert_invalid_runtime_config(config);

    config = RuntimeConfig{};
    config.attachment.required_hold_seconds = -1.0F;
    assert_invalid_runtime_config(config);

    config = RuntimeConfig{};
    config.carry.search_interval_seconds = 0.0F;
    assert_invalid_runtime_config(config);
}

void test_exception_after_reservation_rolls_back_and_can_retry() {
    using namespace interaction;
    RuntimeFixture fixture = make_runtime_fixture();
    InteractionRuntime runtime(
        fixture.database, fixture.features, fixture.registry, RuntimeConfig{});
    RuntimeOutput output = runtime.update(interact_input(
        fixture.locomotion, fixture.request));
    assert(output.diagnostics.state == RuntimeState::Preflight);

    const std::vector<float> offsets = fixture.features.offsets;
    fixture.features.offsets.clear();
    bool threw = false;
    try {
        (void)advance(runtime, fixture.locomotion);
    } catch (const std::out_of_range&) {
        threw = true;
    }
    assert(threw);
    assert(runtime.state() == RuntimeState::Preflight);
    assert_free(fixture.registry, fixture.request.target);

    fixture.features.offsets = offsets;
    output = advance(runtime, fixture.locomotion);
    assert(output.diagnostics.state == RuntimeState::Align);
    assert(output.diagnostics.result == ResultCode::Accepted);
    assert(fixture.registry.validate(
        fixture.request.target, fixture.request.request_id));
}

void test_high_cost_is_rejected_as_poor_match_and_stays_terminal() {
    using namespace interaction;
    RuntimeFixture fixture = high_cost_fixture();
    InteractionRuntime runtime(
        fixture.database, fixture.features, fixture.registry, RuntimeConfig{});
    RuntimeOutput output = runtime.update(interact_input(
        fixture.locomotion, fixture.request));
    assert(output.diagnostics.state == RuntimeState::Preflight);
    output = advance(runtime, fixture.locomotion);
    assert(output.diagnostics.state == RuntimeState::Locomotion);
    assert(output.diagnostics.result == ResultCode::Rejected);
    assert(output.diagnostics.reason == Reason::PoorMatch);
    assert_free(fixture.registry, fixture.request.target);

    const RuntimeDiagnostics terminal = output.diagnostics;
    output = advance(runtime, fixture.locomotion);
    assert(output.diagnostics.state == terminal.state);
    assert(output.diagnostics.result == terminal.result);
    assert(output.diagnostics.reason == terminal.reason);
    assert(output.diagnostics.target == terminal.target);
    assert(output.diagnostics.affordance_id == terminal.affordance_id);
    assert(output.diagnostics.clip == terminal.clip);
    assert(output.diagnostics.frame == terminal.frame);
    assert(!output.owns_pose && !output.suppress_steering);

    assert(run_rejected(high_cost_fixture()).reason == Reason::PoorMatch);
}

void test_contact_failure_rejection_is_terminal_and_never_owns_pose() {
    using namespace interaction;
    RuntimeFixture fixture = contact_failure_fixture();
    RuntimeConfig config{};
    config.ik.accepted_position_m =
        config.ik.maximum_request_position_m;
    InteractionRuntime runtime(
        fixture.database, fixture.features, fixture.registry, config);
    RuntimeOutput output = runtime.update(interact_input(
        fixture.locomotion, fixture.request));
    assert(output.diagnostics.state == RuntimeState::Preflight);
    output = advance(runtime, fixture.locomotion);
    assert(output.diagnostics.state == RuntimeState::Locomotion);
    assert(output.diagnostics.result == ResultCode::Rejected);
    assert(output.diagnostics.reason == Reason::CorrectionLimit);
    assert(!output.diagnostics.attached);
    assert(!output.owns_pose && !output.suppress_steering);
    assert_free(fixture.registry, fixture.request.target);

    const RuntimeDiagnostics terminal = output.diagnostics;
    output = advance(runtime, fixture.locomotion);
    assert(output.diagnostics.state == terminal.state);
    assert(output.diagnostics.result == terminal.result);
    assert(output.diagnostics.reason == terminal.reason);
    assert(!output.diagnostics.attached);
    assert(!output.owns_pose && !output.suppress_steering);
}

void test_hold_extends_the_frozen_final_pose_without_resetting_timer() {
    using namespace interaction;
    RuntimeFixture fixture = make_runtime_fixture();
    RuntimeConfig config{};
    config.attachment.required_hold_seconds = 2.0F;
    InteractionRuntime runtime(
        fixture.database, fixture.features, fixture.registry, config);
    RuntimeOutput output = runtime.update(interact_input(
        fixture.locomotion, fixture.request));
    assert(output.diagnostics.state == RuntimeState::Preflight);
    output = advance(runtime, fixture.locomotion);
    assert(output.diagnostics.state == RuntimeState::Align);
    output = advance_until(
        runtime,
        fixture.locomotion,
        RuntimeState::PickupReplay,
        RuntimeState::Align);
    output = advance_until(
        runtime,
        fixture.locomotion,
        RuntimeState::Hold,
        RuntimeState::PickupReplay);

    int hold_updates = 0;
    bool saw_final_frame = false;
    bool saw_frozen_extension = false;
    Pose final_pose{};
    while (output.diagnostics.state == RuntimeState::Hold) {
        if (output.diagnostics.frame ==
                fixture.database.range_stops.at(1) - 1 &&
            output.diagnostics.state != RuntimeState::Locomotion) {
            if (!saw_final_frame) {
                final_pose = output.pose;
                saw_final_frame = true;
            } else {
                assert(exact(output.pose, final_pose));
                saw_frozen_extension = true;
            }
        }
        ++hold_updates;
        assert(hold_updates < kMaximumUpdates);
        output = advance(runtime, fixture.locomotion);
    }
    assert(output.diagnostics.state == RuntimeState::Carry);
    assert(output.diagnostics.result == ResultCode::Succeeded);
    assert(saw_final_frame && saw_frozen_extension);
    assert(static_cast<float>(hold_updates) * kDt <
        config.attachment.required_hold_seconds);
}

void test_post_attach_failure_preserves_object_and_frees_registry() {
    using namespace interaction;
    RuntimeFixture fixture = post_attach_contact_loss_fixture();
    const TargetHandle original_handle = fixture.request.target;
    InteractionRuntime runtime(
        fixture.database, fixture.features, fixture.registry, RuntimeConfig{});
    RuntimeOutput output = runtime.update(interact_input(
        fixture.locomotion, fixture.request));
    assert(output.diagnostics.state == RuntimeState::Preflight);
    output = advance(runtime, fixture.locomotion);
    assert(output.diagnostics.state == RuntimeState::Align);

    bool saw_attached = false;
    bool saw_failure = false;
    bool saw_final_owned_frame = false;
    Transform last_attached_object{};
    for (int update = 0; update < kMaximumUpdates; ++update) {
        output = advance(runtime, fixture.locomotion);
        if (output.diagnostics.attached) {
            assert(!saw_failure);
            saw_attached = true;
            last_attached_object = output.object_world;
        }
        if (output.diagnostics.result == ResultCode::Failed) {
            saw_failure = true;
            assert(output.diagnostics.reason == Reason::LostContact);
            assert(!output.diagnostics.attached);
        }
        if (output.diagnostics.frame ==
                fixture.database.range_stops.at(1) - 1 &&
            output.diagnostics.state != RuntimeState::Locomotion) {
            assert(output.owns_pose);
            saw_final_owned_frame = true;
        }
        if (output.diagnostics.state == RuntimeState::Locomotion) break;
    }
    assert(output.diagnostics.state == RuntimeState::Locomotion);
    assert(output.diagnostics.result == ResultCode::Failed);
    assert(output.diagnostics.reason == Reason::LostContact);
    assert(saw_attached && saw_failure && saw_final_owned_frame);
    assert(!output.diagnostics.attached);
    assert(exact(output.object_world, last_attached_object));

    const std::optional<TargetHandle> replacement =
        fixture.registry.resolve_single_target(
            last_attached_object.position, 1.0F);
    assert(replacement.has_value());
    assert(replacement->id == original_handle.id);
    assert(replacement->generation == original_handle.generation + 1U);
    const InteractionTarget* target = fixture.registry.find(*replacement);
    assert(target != nullptr);
    assert(target->state == ObjectState::Free);
    assert(target->owner_request == 0U);
    assert(exact(target->object_world, last_attached_object));
}

void test_post_attach_target_change_never_clobbers_a_newer_generation() {
    using namespace interaction;
    RuntimeFixture fixture = make_runtime_fixture();
    InteractionRuntime runtime(
        fixture.database, fixture.features, fixture.registry, RuntimeConfig{});
    RuntimeOutput output = runtime.update(interact_input(
        fixture.locomotion, fixture.request));
    assert(output.diagnostics.state == RuntimeState::Preflight);
    output = advance(runtime, fixture.locomotion);
    assert(output.diagnostics.state == RuntimeState::Align);
    output = advance_until(
        runtime,
        fixture.locomotion,
        RuntimeState::PickupReplay,
        RuntimeState::Align);
    output = advance_until(
        runtime,
        fixture.locomotion,
        RuntimeState::Hold,
        RuntimeState::PickupReplay);
    assert(output.diagnostics.attached);

    Transform authoritative = fixture.original_object_world;
    authoritative.position.x += 0.30F;
    authoritative.position.z += 0.20F;
    const TargetHandle newer = fixture.registry.replace_pose(
        fixture.request.target.id, authoritative);
    assert(newer.generation == fixture.request.target.generation + 1U);

    bool saw_final_owned_frame = false;
    for (int update = 0; update < kMaximumUpdates; ++update) {
        output = advance(runtime, fixture.locomotion);
        if (output.diagnostics.frame ==
                fixture.database.range_stops.at(1) - 1 &&
            output.diagnostics.state != RuntimeState::Locomotion) {
            assert(output.owns_pose);
            saw_final_owned_frame = true;
        }
        if (output.diagnostics.state == RuntimeState::Locomotion) break;
    }
    assert(output.diagnostics.state == RuntimeState::Locomotion);
    assert(output.diagnostics.result == ResultCode::Failed);
    assert(output.diagnostics.reason == Reason::TargetChanged);
    assert(saw_final_owned_frame);

    const InteractionTarget* preserved = fixture.registry.find(newer);
    assert(preserved != nullptr);
    assert(preserved->state == ObjectState::Free);
    assert(preserved->owner_request == 0U);
    assert(exact(preserved->object_world, authoritative));
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 2 &&
        std::strcmp(argv[1], "--pick-map-fast-math-canary") == 0) {
        test_pick_snapshot_map_rejects_every_nonfinite_or_nonunit_input();
        return 0;
    }
    assert(argc == 1);
    test_pick_snapshot_map_is_a_rigid_planar_world_map();
    test_pick_snapshot_map_preserves_local_channels_and_input();
    test_pick_snapshot_map_rejects_every_nonfinite_or_nonunit_input();
    test_pick_snapshot_fingerprint_is_fieldwise_and_canonical();
    test_pick_preview_public_api_reports_path_and_match_separately();
    test_pick_preview_rejects_invalid_runtime_target_and_root_inputs();
    test_pick_preview_is_deterministic_and_const_on_every_outcome();
    test_pick_preview_matches_normal_preflight_for_same_realized_snapshot();
    test_pickup_source_provenance_preview_preflight_and_registry_join();
    test_pick_preview_never_reserves_or_constructs_request_authority();
    test_pick_preview_rejection_reason_mapping_is_exact();
    test_frozen_public_contract_and_defaults();
    test_runtime_preview_exposes_precomputed_provenance_boundary();
    test_place_preflight_and_replay_preserve_precomputed_provenance();
    test_place_preview_and_collapsed_success_lifecycle();
    test_runtime_validates_complete_place_and_ik_configuration();
    test_runtime_forwards_nondefault_place_config_exactly();
    test_runtime_ik_identity_binds_every_scalar_and_iteration();
    test_place_preflight_rejections_preserve_frozen_carry();
    test_carry_rejects_authored_slot_order_change();
    test_place_preflight_rejects_authored_slot_metadata_change();
    test_post_begin_place_failures_reconstruct_fresh_carry();
    test_place_cancellation_boundaries_for_recorded_and_reverse();
    test_cancel_on_release_update_preserves_release_for_both_modes();
    test_reset_and_duplicate_edges_are_ignored_in_every_place_state();
    test_recorded_and_reverse_place_diagnostics_are_deterministic();
    test_disabled_is_permanent_pose_passthrough();
    test_enabled_runtime_rejects_noncanonical_dt_before_mutation();
    test_noncanonical_dt_cannot_mutate_owned_runtime_state();
    test_interact_without_explicit_request_is_rejected_after_preflight();
    test_success_order_carry_and_reset();
    test_layered_carry_publishes_inactive_arm_authority_transactionally();
    test_recorded_and_weighted_carry_never_publish_inactive_arm_authority();
    test_hand_constraint_weight_tracks_authored_reach_and_attachment();
    test_place_hand_constraint_weight_is_full_until_release();
    test_hand_constraint_weight_resets_on_cancel_failure_and_reset();
    test_carry_reset_preserves_a_newer_authoritative_generation();
    test_carry_update_preserves_a_newer_authoritative_generation();
    test_align_cancel_releases_reservation();
    test_entry_blends_the_whole_pose_for_quarter_second();
    test_entry_blend_continues_after_early_commit();
    test_cancel_after_commit_before_contact_releases_pickup();
    test_cancel_after_contact_preserves_attached_pickup();
    test_commit_at_contact_preserves_the_one_shot_crossing();
    test_nonunit_playback_speed_commits_by_source_contact();
    test_realized_transition_rejects_contact_that_cannot_attach();
    test_realized_transition_rejects_invalid_reconstructed_object();
    test_realized_contact_reason_adapter_preserves_matcher_categories();
    test_canonical_updates_reject_invalid_contact_before_commit();
    test_canonical_updates_cannot_skip_post_attach_contact_loss();
    test_pickup_success_precedes_later_canonical_contact_loss();
    test_hold_interval_precedes_its_canonical_stop_frame_event();
    test_blocked_reach_transition_falls_back_to_safe_approach_entry();
    test_nonunit_fractional_wall_endpoint_is_certified_before_commit();
    test_realized_post_ik_table_sweep_is_rejected_in_preflight();
    test_realized_post_ik_object_sweep_is_rejected_in_preflight();
    test_curved_precontact_blocked_reach_falls_back_to_safe_approach();
    test_runtime_clearance_backstop_rejects_post_certification_change();
    test_stale_generation_is_rejected();
    test_align_rejects_profile_or_bounds_snapshot_changes();
    test_failed_reservation_never_releases_an_existing_owner();
    test_runtime_config_is_validated_before_reservation();
    test_exception_after_reservation_rolls_back_and_can_retry();
    test_high_cost_is_rejected_as_poor_match_and_stays_terminal();
    test_contact_failure_rejection_is_terminal_and_never_owns_pose();
    test_hold_extends_the_frozen_final_pose_without_resetting_timer();
    test_post_attach_failure_preserves_object_and_frees_registry();
    test_post_attach_target_change_never_clobbers_a_newer_generation();
}
