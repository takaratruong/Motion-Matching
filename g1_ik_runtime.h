#pragma once

#include "g1_clearance.h"
#include "g1_footprint_runtime.h"
#include "g1_ik.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

enum G1IkStopReason
{
    G1IkStopNone = 0,
    G1IkStopFootprintBlocked,
    G1IkStopFootprintOutsideDomain,
    G1IkStopFootprintBudgetExceeded,
    G1IkStopLandingPatchUnavailable,
    G1IkStopTargetUnreachable,
    G1IkStopNoSwingCandidate,
    G1IkStopPoseClearanceRejected,
};

static constexpr uint32_t G1SwingLiftCandidateCount = 41;
static constexpr uint32_t G1SwingNoCandidate = UINT32_MAX;

struct G1SwingCandidateDiagnostic
{
    uint32_t candidate_index = G1SwingNoCandidate;
    uint32_t lift_bits = 0;
    uint32_t materialized_command_y_bits = 0;
    uint32_t actual_sphere_center_bits[4][3] = {};
    G1ClearanceStatus clearance_status = G1ClearanceInvalidInput;
    bool controller_constraints_passed = false;
    bool clearance_certified = false;
    double lower_margin_m = 0.0;
    double witness_upper_margin_m = 0.0;
    G1ClearanceWork clearance_work;
};

struct G1SwingSelectionDiagnostic
{
    uint32_t candidates_evaluated = 0;
    uint32_t selected_index = G1SwingNoCandidate;
    G1SwingCandidateDiagnostic selected;
    G1ClearanceWork total_clearance_work;
};

struct G1FootIkState
{
    G1FootLockState lock;
    G1SwingHistory swing;
    vec3 baseline_sole_normal;
};

struct G1IkState
{
    bool initialized = false;
    G1FootIkState feet[2];
};

struct G1FootFrameResult
{
    bool recorded_contact = false;
    G1FootTarget target;
    G1SwingSelectionDiagnostic swing_selection;
    G1SwingClearanceValidation defensive_swing;
    G1LegSolveResult position;
    G1FootOrientationResult orientation;
};

struct G1IkFrameResult
{
    bool applied = false;
    bool safe_stop_requested = false;
    G1IkStopReason stop_reason = G1IkStopNone;
    float max_correction_radians = 0.0f;
    G1FootFrameResult feet[2];
};

struct G1IkFrameTransaction
{
    bool initialized = false;
    uint32_t next_foot = 0;
    G1IkState candidate_state;
    G1IkFrameResult candidate_result;
    G1LegIterationProvenance staged_iteration_provenance[2] = {
        G1LegIterationNone,
        G1LegIterationNone,
    };
};

static inline bool g1_ik_runtime_iteration_transcript_matches(
    const G1IkFrameTransaction& transaction)
{
    if (transaction.next_foot > 2U) return false;
    for (uint32_t foot = 0; foot < 2U; ++foot) {
        const G1LegSolveResult& position =
            transaction.candidate_result.feet[foot].position;
        const G1LegIterationProvenance expected = position.applied
            ? position.iteration_provenance
            : G1LegIterationNone;
        if (transaction.staged_iteration_provenance[foot] != expected ||
            (foot >= transaction.next_foot &&
             transaction.staged_iteration_provenance[foot] !=
                 G1LegIterationNone)) {
            return false;
        }
    }
    return true;
}

enum G1IkRejectionCheckpoint
{
    G1IkRejectionAfterBegin = 0,
    G1IkRejectionAfterFoot0,
    G1IkRejectionAfterFoot1,
};

struct G1IkSafeStopHandoff
{
    vec3 applied_velocity;
    bool cancel_planar_inertia = false;
    bool force_search = false;
};

static constexpr uint32_t G1SwingLiftCandidateBits[
    G1SwingLiftCandidateCount] = {
    0x00000000u, 0x3b03126fu, 0x3b83126fu, 0x3bc49ba6u,
    0x3c03126fu, 0x3c23d70au, 0x3c449ba6u, 0x3c656042u,
    0x3c83126fu, 0x3c9374bcu, 0x3ca3d70au, 0x3cb43958u,
    0x3cc49ba6u, 0x3cd4fdf4u, 0x3ce56042u, 0x3cf5c28fu,
    0x3d03126fu, 0x3d0b4396u, 0x3d1374bcu, 0x3d1ba5e3u,
    G1_Exact25HzBits, 0x3d2c0831u, 0x3d343958u, 0x3d3c6a7fu,
    0x3d449ba6u, 0x3d4ccccdu, 0x3d54fdf4u, 0x3d5d2f1bu,
    0x3d656042u, 0x3d6d9168u, 0x3d75c28fu, 0x3d7df3b6u,
    0x3d83126fu, 0x3d872b02u, 0x3d8b4396u, 0x3d8f5c29u,
    0x3d9374bcu, 0x3d978d50u, 0x3d9ba5e3u, 0x3d9fbe77u,
    0x3da3d70au,
};

static_assert(sizeof(float) == sizeof(uint32_t),
              "G1 IK lift ladder requires binary32 storage");

static constexpr bool g1_ik_runtime_lift_ladder_is_strictly_ordered()
{
    if (G1SwingLiftCandidateBits[0] != UINT32_C(0x00000000) ||
        G1SwingLiftCandidateBits[G1SwingLiftCandidateCount - 1] !=
            UINT32_C(0x3da3d70a)) {
        return false;
    }
    for (uint32_t candidate = 1;
         candidate < G1SwingLiftCandidateCount;
         ++candidate) {
        if (G1SwingLiftCandidateBits[candidate - 1] >=
            G1SwingLiftCandidateBits[candidate]) {
            return false;
        }
    }
    return true;
}

static_assert(g1_ik_runtime_lift_ladder_is_strictly_ordered(),
              "all 41 G1 IK lift words are immutable and strictly ordered");

static inline bool g1_ik_runtime_fail(
    char* error, int error_capacity, const char* message)
{
    return g1_ik_error(error, error_capacity, "%s", message);
}

static inline uint32_t g1_ik_runtime_float_bits(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static inline bool g1_ik_runtime_work_add(
    G1ClearanceWork& output,
    const G1ClearanceWork& incoming)
{
    uint32_t G1ClearanceWork::* const members[] = {
        &G1ClearanceWork::point_queries,
        &G1ClearanceWork::cells_visited,
        &G1ClearanceWork::primitive_triangle_pairs,
        &G1ClearanceWork::face_patches,
        &G1ClearanceWork::candidate_tests,
        &G1ClearanceWork::subdivision_nodes
    };
    G1ClearanceWork candidate = output;
    for (const auto member : members) {
        if (incoming.*member > UINT32_MAX - candidate.*member) {
            return false;
        }
        candidate.*member += incoming.*member;
    }
    output = candidate;
    return true;
}

static inline bool g1_ik_runtime_history_is_valid(
    const G1SwingHistory& history)
{
    if (!history.initialized) return false;
    for (int probe = 0; probe < 4; ++probe) {
        if (!g1_ik_vec3_is_runtime_value(
                history.previous_sphere_centers[probe])) {
            return false;
        }
    }
    return true;
}

static inline bool g1_ik_runtime_state_is_valid(const G1IkState& state)
{
    if (!state.initialized) return false;
    for (int foot = 0; foot < 2; ++foot) {
        if (!g1_foot_lock_state_is_valid(state.feet[foot].lock) ||
            !g1_ik_runtime_history_is_valid(state.feet[foot].swing) ||
            !g1_ik_surface_normal_is_valid(
                state.feet[foot].baseline_sole_normal)) {
            return false;
        }
    }
    return true;
}

static inline bool g1_ik_runtime_pose_shapes_are_exact(
    const slice1d<vec3> positions,
    const slice1d<quat> rotations,
    const slice1d<int> parents)
{
    return positions.size == G1_BoneCount && positions.data != NULL &&
           rotations.size == G1_BoneCount && rotations.data != NULL &&
           parents.size == G1_BoneCount && parents.data != NULL;
}

static inline bool g1_ik_runtime_arrays_are_exact(
    const array1d<vec3>& positions,
    const array1d<quat>& rotations)
{
    return positions.size == G1_BoneCount && positions.data != NULL &&
           rotations.size == G1_BoneCount && rotations.data != NULL;
}

static inline bool g1_ik_runtime_pose_values_are_valid(
    const slice1d<vec3> positions,
    const slice1d<quat> rotations)
{
    for (int bone = 0; bone < G1_BoneCount; ++bone) {
        if (!g1_ik_vec3_is_runtime_value(positions.data[bone]) ||
            !ik_quat_is_unit(rotations.data[bone])) {
            return false;
        }
    }
    return true;
}

static inline bool g1_ik_runtime_pose_ranges_are_disjoint(
    const array1d<vec3>& output_positions,
    const array1d<quat>& output_rotations,
    const slice1d<vec3> input_positions,
    const slice1d<quat> input_rotations,
    const slice1d<int> parents)
{
    const std::size_t position_bytes =
        static_cast<std::size_t>(G1_BoneCount) * sizeof(vec3);
    const std::size_t rotation_bytes =
        static_cast<std::size_t>(G1_BoneCount) * sizeof(quat);
    const std::size_t parent_bytes =
        static_cast<std::size_t>(G1_BoneCount) * sizeof(int);
    return !g1_ik_memory_ranges_overlap(
               output_positions.data, position_bytes,
               output_rotations.data, rotation_bytes) &&
           !g1_ik_memory_ranges_overlap(
               output_positions.data, position_bytes,
               input_positions.data, position_bytes) &&
           !g1_ik_memory_ranges_overlap(
               output_positions.data, position_bytes,
               input_rotations.data, rotation_bytes) &&
           !g1_ik_memory_ranges_overlap(
               output_positions.data, position_bytes,
               parents.data, parent_bytes) &&
           !g1_ik_memory_ranges_overlap(
               output_rotations.data, rotation_bytes,
               input_positions.data, position_bytes) &&
           !g1_ik_memory_ranges_overlap(
               output_rotations.data, rotation_bytes,
               input_rotations.data, rotation_bytes) &&
           !g1_ik_memory_ranges_overlap(
               output_rotations.data, rotation_bytes,
               parents.data, parent_bytes);
}

static inline bool
g1_ik_runtime_compute_foot_centers(
    vec3 output[4],
    const slice1d<vec3> global_positions,
    const slice1d<quat> global_rotations,
    const G1LegConfig& config)
{
    vec3 sole_points[4] = {};
    // The footprint observer and IK preflight intentionally share this
    // no-clone materialization boundary.  Exact provenance must not depend
    // on the optimizer choosing different reassociation for the two owners.
    return g1_footprint_materialize_current_geometry(
        output,
        sole_points,
        global_positions(config.contact),
        global_rotations(config.contact),
        config);
}

static inline const char* g1_ik_stop_reason_name(G1IkStopReason reason)
{
    switch (reason) {
    case G1IkStopNone: return "none";
    case G1IkStopFootprintBlocked: return "footprint-blocked";
    case G1IkStopFootprintOutsideDomain:
        return "footprint-outside-domain";
    case G1IkStopFootprintBudgetExceeded:
        return "footprint-budget-exceeded";
    case G1IkStopLandingPatchUnavailable:
        return "landing-patch-unavailable";
    case G1IkStopTargetUnreachable: return "target-unreachable";
    case G1IkStopNoSwingCandidate: return "no-swing-candidate";
    case G1IkStopPoseClearanceRejected:
        return "pose-clearance-rejected";
    default: return "invalid";
    }
}

static inline bool g1_ik_safe_stop_handoff(
    G1IkSafeStopHandoff& output,
    bool latched,
    vec3 requested_velocity,
    char* error,
    int error_capacity)
{
    if (!g1_ik_vec3_is_runtime_value(requested_velocity)) {
        return g1_ik_runtime_fail(
            error, error_capacity,
            "G1 IK safe-stop handoff requires runtime velocity");
    }
    G1IkSafeStopHandoff candidate = {};
    candidate.applied_velocity = requested_velocity;
    if (latched) {
        candidate.applied_velocity.x = 0.0f;
        candidate.applied_velocity.z = 0.0f;
        candidate.cancel_planar_inertia = true;
        candidate.force_search = true;
    }
    output = candidate;
    return true;
}

static inline bool g1_ik_state_reset(
    G1IkState& output,
    const slice1d<vec3> local_positions,
    const slice1d<quat> local_rotations,
    const slice1d<int> bone_parents,
    char* error,
    int error_capacity)
{
    if (error_capacity < 0 ||
        !g1_ik_runtime_pose_shapes_are_exact(
            local_positions, local_rotations, bone_parents) ||
        !g1_ik_runtime_pose_values_are_valid(
            local_positions, local_rotations) ||
        !g1_ik_parent_topology_validate(
            bone_parents, error, error_capacity) ||
        !g1_foot_runtime_config_validate(
            g1_left_leg_config(), error, error_capacity) ||
        !g1_foot_runtime_config_validate(
            g1_right_leg_config(), error, error_capacity)) {
        return false;
    }

    vec3 global_positions_storage[G1_BoneCount];
    quat global_rotations_storage[G1_BoneCount];
    const slice1d<vec3> global_positions(
        G1_BoneCount, global_positions_storage);
    const slice1d<quat> global_rotations(
        G1_BoneCount, global_rotations_storage);
    if (!g1_ik_checked_forward_kinematics(
            global_positions, global_rotations,
            local_positions, local_rotations, bone_parents,
            error, error_capacity)) {
        return false;
    }

    const G1LegConfig configs[2] = {
        g1_left_leg_config(), g1_right_leg_config()
    };
    G1IkState candidate = {};
    candidate.initialized = true;
    for (int foot = 0; foot < 2; ++foot) {
        vec3 centers[4] = {};
        vec3 sole_center;
        vec3 sole_normal;
        if (!g1_ik_runtime_compute_foot_centers(
                centers, global_positions, global_rotations,
                configs[foot]) ||
            !g1_ik_checked_physical_sole_centroid(
                sole_center,
                global_positions(configs[foot].contact),
                global_rotations(configs[foot].contact),
                configs[foot]) ||
            !ik_checked_quat_rotate(
                sole_normal,
                global_rotations(configs[foot].contact),
                configs[foot].sole_normal_local) ||
            !g1_ik_surface_normal_is_valid(sole_normal) ||
            !g1_foot_lock_reset(
                candidate.feet[foot].lock,
                sole_center,
                error, error_capacity) ||
            !g1_swing_history_reset(
                candidate.feet[foot].swing,
                centers, error, error_capacity)) {
            return false;
        }
        candidate.feet[foot].baseline_sole_normal = sole_normal;
    }
    if (!g1_ik_runtime_state_is_valid(candidate)) {
        return g1_ik_runtime_fail(
            error, error_capacity,
            "G1 IK state reset produced invalid complete state");
    }
    output = candidate;
    return true;
}

static inline bool g1_ik_runtime_surface_sample_is_valid(
    const G1SurfaceSample& sample)
{
    return g1_ik_float_is_runtime_value(sample.height) &&
           g1_ik_surface_normal_is_valid(sample.normal);
}

static inline bool g1_ik_runtime_surface_status_is_valid(
    G1SurfaceQueryStatus status)
{
    return status == G1SurfaceQueryValid ||
           status == G1SurfaceQueryOutside ||
           status == G1SurfaceQueryInvalid;
}

static inline bool g1_ik_runtime_walkability_reason_is_valid(
    walkability_reason reason)
{
    return reason == walkability_clear ||
           reason == walkability_blocked_cell ||
           reason == walkability_out_of_bounds ||
           reason == walkability_nonfinite;
}

static inline bool g1_ik_runtime_class_is_valid(int value)
{
    return value >= 0 && value <= 2;
}

static inline bool g1_ik_runtime_footprint_validate(
    const G1FootprintObservation& footprint,
    const slice1d<bool> contacts,
    const vec3 current_centers[2][4],
    char* error,
    int error_capacity)
{
    const G1FootprintBudget budget = g1_footprint_budget();
    if (contacts.size != 2 || contacts.data == NULL ||
        !g1_ik_runtime_surface_sample_is_valid(
            footprint.root_surface) ||
        footprint.work.sweeps > budget.maximum_sweeps ||
        footprint.work.surface_queries >
            budget.maximum_surface_queries ||
        footprint.work.node_visits > budget.maximum_node_visits ||
        !g1_ik_runtime_walkability_reason_is_valid(
            footprint.blocked_reason) ||
        (footprint.blocked &&
         footprint.blocked_reason == walkability_clear) ||
        (!footprint.blocked &&
         footprint.blocked_reason != walkability_clear)) {
        return g1_ik_runtime_fail(
            error, error_capacity,
            "G1 IK footprint has invalid common diagnostics");
    }

    for (int foot_index = 0; foot_index < 2; ++foot_index) {
        const G1FootprintFootObservation& foot =
            footprint.feet[foot_index];
        if (foot.current_contact != contacts.data[foot_index] ||
            !g1_ik_runtime_class_is_valid(
                foot.encountered_walkability_class) ||
            !g1_ik_runtime_class_is_valid(
                foot.predicted_landing_walkability_class) ||
            !g1_ik_float_is_runtime_value(
                foot.corridor_minimum_height) ||
            !g1_ik_float_is_runtime_value(
                foot.corridor_maximum_height) ||
            foot.corridor_minimum_height >
                foot.corridor_maximum_height ||
            !terrain_double_is_finite(foot.maximum_root_split_m) ||
            foot.maximum_root_split_m < 0.0 ||
            !terrain_double_is_finite(
                foot.landing_patch_maximum_residual_m) ||
            foot.landing_patch_maximum_residual_m < 0.0) {
            return g1_ik_runtime_fail(
                error, error_capacity,
                "G1 IK footprint has invalid per-foot diagnostics");
        }

        for (int probe_index = 0; probe_index < 4; ++probe_index) {
            const G1FootprintProbe& probe = foot.probes[probe_index];
            if (!g1_ik_vec3_is_runtime_value(
                    probe.current_sphere_center) ||
                !g1_ik_vec3_bits_equal(
                    probe.current_sphere_center,
                    current_centers[foot_index][probe_index]) ||
                !g1_ik_vec3_is_runtime_value(
                    probe.current_sole_point) ||
                !g1_ik_runtime_surface_sample_is_valid(
                    probe.current_surface) ||
                !g1_ik_float_is_runtime_value(
                    probe.corridor_minimum_height) ||
                !g1_ik_float_is_runtime_value(
                    probe.corridor_maximum_height) ||
                probe.corridor_minimum_height >
                    probe.corridor_maximum_height ||
                !g1_ik_runtime_class_is_valid(
                    probe.encountered_walkability_class)) {
                return g1_ik_runtime_fail(
                    error, error_capacity,
                    "G1 IK footprint current probe does not match FK");
            }
            for (int sample = 0;
                 sample < G1CommandTrajectorySampleCount;
                 ++sample) {
                const G1SurfaceQueryStatus status =
                    probe.predicted_surface_status[sample];
                if (!g1_ik_vec3_is_runtime_value(
                        probe.predicted_sphere_centers[sample]) ||
                    !g1_ik_vec3_is_runtime_value(
                        probe.predicted_sole_points[sample]) ||
                    !g1_ik_runtime_surface_status_is_valid(status) ||
                    (status == G1SurfaceQueryValid &&
                     !g1_ik_runtime_surface_sample_is_valid(
                         probe.predicted_surfaces[sample]))) {
                    return g1_ik_runtime_fail(
                        error, error_capacity,
                        "G1 IK footprint predicted probe is malformed");
                }
            }
        }

        if (!foot.landing_expected) {
            if (foot.landing_patch_ready ||
                foot.landing_sample != UINT32_MAX) {
                return g1_ik_runtime_fail(
                    error, error_capacity,
                    "G1 IK footprint has an unexpected landing patch");
            }
            continue;
        }
        if (foot.landing_sample >= G1CommandTrajectorySampleCount ||
            !g1_ik_vec3_is_runtime_value(
                foot.predicted_landing_sole_center) ||
            !g1_ik_runtime_surface_status_is_valid(
                foot.predicted_landing_surface_status)) {
            return g1_ik_runtime_fail(
                error, error_capacity,
                "G1 IK footprint landing prediction is malformed");
        }
        if (foot.landing_patch_ready &&
            (foot.predicted_landing_surface_status !=
                 G1SurfaceQueryValid ||
             foot.predicted_landing_walkability_class != 1 ||
             !g1_ik_runtime_surface_sample_is_valid(
                 foot.predicted_landing_surface))) {
            return g1_ik_runtime_fail(
                error, error_capacity,
                "G1 IK footprint ready landing patch is invalid");
        }
    }
    return true;
}

static inline void g1_ik_runtime_request_safe_stop(
    G1IkFrameTransaction& transaction,
    G1IkStopReason reason)
{
    transaction.candidate_result.applied = false;
    transaction.candidate_result.safe_stop_requested = true;
    transaction.candidate_result.stop_reason = reason;
}

static inline bool g1_ik_runtime_validate_landing_lookahead(
    const G1FootTarget& target,
    const G1FootprintFootObservation& footprint,
    const G1LegConfig& config,
    char* error,
    int error_capacity)
{
    if (!g1_foot_target_is_valid(target) ||
        !g1_foot_runtime_config_validate(
            config, error, error_capacity) ||
        !footprint.landing_expected ||
        !footprint.landing_patch_ready ||
        footprint.landing_sample >=
            G1CommandTrajectorySampleCount ||
        !g1_ik_vec3_is_runtime_value(
            footprint.predicted_landing_sole_center) ||
        footprint.predicted_landing_surface_status !=
            G1SurfaceQueryValid ||
        footprint.predicted_landing_walkability_class != 1 ||
        !g1_ik_runtime_surface_sample_is_valid(
            footprint.predicted_landing_surface)) {
        return g1_ik_runtime_fail(
            error, error_capacity,
            "G1 IK ready landing lookahead is invalid");
    }
    return true;
}

static inline bool g1_ik_frame_begin(
    G1IkFrameTransaction& transaction,
    array1d<vec3>& scratch_positions,
    array1d<quat>& scratch_rotations,
    const G1IkState& state,
    const slice1d<vec3> baseline_positions,
    const slice1d<quat> baseline_rotations,
    const slice1d<int> bone_parents,
    const slice1d<bool> contacts,
    const heightfield& field,
    const G1FootprintObservation& footprint,
    bool enabled,
    float dt,
    char* error,
    int error_capacity)
{
    if (error_capacity < 0 ||
        !g1_dt_is_exact_25_hz(dt) ||
        field.version != 2 ||
        !terrain_heightfield_is_queryable(field) ||
        !g1_ik_runtime_state_is_valid(state) ||
        !g1_ik_runtime_pose_shapes_are_exact(
            baseline_positions, baseline_rotations, bone_parents) ||
        !g1_ik_runtime_arrays_are_exact(
            scratch_positions, scratch_rotations) ||
        !g1_ik_runtime_pose_values_are_valid(
            baseline_positions, baseline_rotations) ||
        !g1_ik_runtime_pose_ranges_are_disjoint(
            scratch_positions, scratch_rotations,
            baseline_positions, baseline_rotations, bone_parents) ||
        !g1_ik_parent_topology_validate(
            bone_parents, error, error_capacity) ||
        contacts.size != 2 || contacts.data == NULL) {
        return false;
    }

    vec3 global_positions_storage[G1_BoneCount];
    quat global_rotations_storage[G1_BoneCount];
    const slice1d<vec3> global_positions(
        G1_BoneCount, global_positions_storage);
    const slice1d<quat> global_rotations(
        G1_BoneCount, global_rotations_storage);
    if (!g1_ik_checked_forward_kinematics(
            global_positions, global_rotations,
            baseline_positions, baseline_rotations, bone_parents,
            error, error_capacity)) {
        return false;
    }

    const G1LegConfig configs[2] = {
        g1_left_leg_config(), g1_right_leg_config()
    };
    vec3 current_centers[2][4] = {};
    for (int foot_index = 0; foot_index < 2; ++foot_index) {
        if (!g1_foot_runtime_config_validate(
                configs[foot_index], error, error_capacity) ||
            !g1_ik_runtime_compute_foot_centers(
                current_centers[foot_index],
                global_positions, global_rotations,
                configs[foot_index])) {
            return false;
        }
    }
    if (!g1_ik_runtime_footprint_validate(
            footprint, contacts, current_centers,
            error, error_capacity)) {
        return false;
    }

    G1IkFrameTransaction candidate = {};
    candidate.initialized = true;
    candidate.candidate_state = state;
    if (!enabled) {
        std::memcpy(
            scratch_positions.data,
            baseline_positions.data,
            static_cast<std::size_t>(G1_BoneCount) * sizeof(vec3));
        std::memcpy(
            scratch_rotations.data,
            baseline_rotations.data,
            static_cast<std::size_t>(G1_BoneCount) * sizeof(quat));
        transaction = candidate;
        return true;
    }

    for (int foot_index = 0; foot_index < 2; ++foot_index) {
        G1FootTarget target = {};
        vec3 current_sole_center;
        vec3 current_sole_normal;
        if (!g1_ik_checked_physical_sole_centroid(
                current_sole_center,
                global_positions(configs[foot_index].contact),
                global_rotations(configs[foot_index].contact),
                configs[foot_index]) ||
            !ik_checked_quat_rotate(
                current_sole_normal,
                global_rotations(configs[foot_index].contact),
                configs[foot_index].sole_normal_local) ||
            !g1_ik_surface_normal_is_valid(
                current_sole_normal)) {
            return g1_ik_runtime_fail(
                error, error_capacity,
                "G1 IK frame could not materialize physical sole pose");
        }
        if (!g1_foot_lock_update(
                candidate.candidate_state.feet[foot_index].lock,
                target,
                field,
                configs[foot_index],
                current_sole_center,
                contacts.data[foot_index],
                dt,
                error,
                error_capacity)) {
            return false;
        }
        target.desired_sole_normal = contacts.data[foot_index]
            ? target.surface.normal
            : current_sole_normal;
        candidate.candidate_state.feet[foot_index]
            .baseline_sole_normal = current_sole_normal;
        if (!g1_foot_target_is_valid(target)) {
            return g1_ik_runtime_fail(
                error, error_capacity,
                "G1 IK frame produced an invalid desired sole normal");
        }
        candidate.candidate_result.feet[foot_index]
            .recorded_contact = contacts.data[foot_index];
        candidate.candidate_result.feet[foot_index].target = target;
    }

    if (footprint.blocked) {
        g1_ik_runtime_request_safe_stop(
            candidate, G1IkStopFootprintBlocked);
    } else {
        for (int foot_index = 0; foot_index < 2; ++foot_index) {
            const G1FootprintFootObservation& foot =
                footprint.feet[foot_index];
            if (contacts.data[foot_index] || !foot.landing_expected) {
                continue;
            }
            if (!foot.landing_patch_ready) {
                g1_ik_runtime_request_safe_stop(
                    candidate,
                    G1IkStopLandingPatchUnavailable);
                break;
            }
            if (!g1_ik_runtime_validate_landing_lookahead(
                    candidate.candidate_result.feet[foot_index]
                        .target,
                    foot,
                    configs[foot_index],
                    error, error_capacity)) {
                return false;
            }
        }
    }

    std::memcpy(
        scratch_positions.data,
        baseline_positions.data,
        static_cast<std::size_t>(G1_BoneCount) * sizeof(vec3));
    std::memcpy(
        scratch_rotations.data,
        baseline_rotations.data,
        static_cast<std::size_t>(G1_BoneCount) * sizeof(quat));
    transaction = candidate;
    return true;
}

struct G1IkRuntimeStagedCandidate
{
    vec3 positions[G1_BoneCount];
    quat rotations[G1_BoneCount];
    vec3 sphere_centers[4];
    G1SwingCandidateDiagnostic diagnostic;
    G1LegSolveResult position;
    G1FootOrientationResult orientation;
    bool passes = false;
};

static inline bool g1_ik_runtime_clearance_status_is_finite_rejection(
    G1ClearanceStatus status)
{
    return status == G1ClearanceOutsideDomain ||
           status == G1ClearanceBudgetExceeded ||
           status == G1ClearanceUncertified;
}

static inline bool g1_ik_runtime_clearance_status_is_fatal(
    G1ClearanceStatus status)
{
    return status == G1ClearanceInvalidInput ||
           status == G1ClearanceInvalidField ||
           status == G1ClearanceArithmeticFailure;
}

static inline bool g1_ik_runtime_controller_constraints_pass(
    const G1LegSolveResult& position,
    const G1FootOrientationResult& orientation,
    vec3 final_contact_origin,
    quat final_contact_rotation,
    const G1FootTarget& target,
    const G1LegConfig& config)
{
    vec3 final_sole_center;
    double sole_residual_precise_m = DBL_MAX;
    float sole_residual_m = FLT_MAX;
    vec3 final_sole_normal;
    vec3 normalized_target_normal;
    vec3 final_forward;
    vec3 frozen_forward;
    double normal_alignment = 0.0;
    double heading_alignment = 0.0;
    const bool endpoint_is_valid =
        g1_foot_target_is_valid(target) &&
        g1_ik_checked_physical_sole_centroid(
            final_sole_center,
            final_contact_origin,
            final_contact_rotation,
            config) &&
        ik_checked_distance_precise(
            sole_residual_precise_m,
            sole_residual_m,
            final_sole_center,
            target.sole_center) &&
        ik_checked_normalize(
            normalized_target_normal,
            target.desired_sole_normal) &&
        ik_checked_quat_rotate(
            final_sole_normal,
            final_contact_rotation,
            config.sole_normal_local) &&
        ik_checked_quat_rotate(
            final_forward,
            final_contact_rotation,
            config.foot_forward_local) &&
        ik_checked_quat_rotate(
            frozen_forward,
            orientation.target_global_rotation,
            config.foot_forward_local) &&
        ik_checked_dot(
            normal_alignment,
            final_sole_normal,
            normalized_target_normal) &&
        ik_checked_dot(
            heading_alignment,
            final_forward,
            frozen_forward);
    return position.applied &&
           position.reachable &&
           !position.correction_limited &&
           !position.safe_stop_requested &&
           g1_ik_contact_residual_is_converged(
               position.contact_residual_m) &&
           orientation.applied &&
           !orientation.correction_limited &&
           !orientation.safe_stop_requested &&
           endpoint_is_valid &&
           g1_ik_contact_residual_is_converged_precise(
               sole_residual_precise_m) &&
           g1_ik_runtime_float_bits(
               position.contact_residual_m) ==
               g1_ik_runtime_float_bits(sole_residual_m) &&
           normal_alignment >= 0.99999 &&
           heading_alignment >= 0.99999;
}

static inline bool g1_ik_runtime_stage_swing_candidate(
    G1IkRuntimeStagedCandidate& output,
    const slice1d<vec3> local_positions,
    const slice1d<quat> baseline_rotations,
    const slice1d<int> bone_parents,
    const G1SwingHistory& history,
    const heightfield& field,
    const G1LegConfig& config,
    const G1FootTarget& base_target,
    uint32_t candidate_index,
    float dt,
    char* error,
    int error_capacity)
{
    if (error_capacity < 0 ||
        candidate_index >= G1SwingLiftCandidateCount ||
        !g1_dt_is_exact_25_hz(dt) ||
        field.version != 2 ||
        !terrain_heightfield_is_queryable(field) ||
        !g1_ik_runtime_history_is_valid(history) ||
        !g1_foot_target_is_valid(base_target) ||
        !g1_ik_runtime_pose_shapes_are_exact(
            local_positions, baseline_rotations, bone_parents) ||
        !g1_ik_runtime_pose_values_are_valid(
            local_positions, baseline_rotations) ||
        !g1_ik_parent_topology_validate(
            bone_parents, error, error_capacity) ||
        !g1_foot_runtime_config_validate(
            config, error, error_capacity) ||
        g1_ik_runtime_float_bits(config.max_swing_lift_m) !=
            G1SwingLiftCandidateBits[
                G1SwingLiftCandidateCount - 1]) {
        return false;
    }

    G1IkRuntimeStagedCandidate candidate = {};
    std::memcpy(
        candidate.positions,
        local_positions.data,
        sizeof(candidate.positions));
    std::memcpy(
        candidate.rotations,
        baseline_rotations.data,
        sizeof(candidate.rotations));
    candidate.diagnostic.candidate_index = candidate_index;
    candidate.diagnostic.lift_bits =
        G1SwingLiftCandidateBits[candidate_index];

    float lift = 0.0f;
    std::memcpy(
        &lift,
        &G1SwingLiftCandidateBits[candidate_index],
        sizeof(lift));
    G1FootTarget staged_target = base_target;
    const G1ClearanceStatus lift_status = g1_apply_swing_lift_y(
        staged_target.sole_center.y,
        base_target.sole_center.y,
        lift,
        error,
        error_capacity);
    if (lift_status != G1ClearanceOk) {
        return false;
    }
    candidate.diagnostic.materialized_command_y_bits =
        g1_ik_runtime_float_bits(staged_target.sole_center.y);

    const slice1d<vec3> staged_positions(
        G1_BoneCount, candidate.positions);
    const slice1d<quat> staged_rotations(
        G1_BoneCount, candidate.rotations);
    if (!g1_apply_named_physical_sole_ik(
            staged_rotations,
            staged_positions,
            baseline_rotations,
            bone_parents,
            config,
            staged_target.sole_center,
            staged_target.desired_sole_normal,
            candidate.position,
            candidate.orientation,
            error,
            error_capacity)) {
        return false;
    }

    vec3 final_global_storage[G1_BoneCount];
    quat final_rotation_storage[G1_BoneCount];
    const slice1d<vec3> final_globals(
        G1_BoneCount, final_global_storage);
    const slice1d<quat> final_global_rotations(
        G1_BoneCount, final_rotation_storage);
    if (!g1_ik_checked_forward_kinematics(
            final_globals,
            final_global_rotations,
            staged_positions,
            staged_rotations,
            bone_parents,
            error,
            error_capacity) ||
        !g1_ik_runtime_compute_foot_centers(
            candidate.sphere_centers,
            final_globals,
            final_global_rotations,
            config)) {
        return false;
    }

    candidate.diagnostic.controller_constraints_passed =
        g1_ik_runtime_controller_constraints_pass(
            candidate.position,
            candidate.orientation,
            final_globals(config.contact),
            final_global_rotations(config.contact),
            staged_target,
            config);

    for (int probe = 0; probe < 4; ++probe) {
        candidate.diagnostic.actual_sphere_center_bits[probe][0] =
            g1_ik_runtime_float_bits(
                candidate.sphere_centers[probe].x);
        candidate.diagnostic.actual_sphere_center_bits[probe][1] =
            g1_ik_runtime_float_bits(
                candidate.sphere_centers[probe].y);
        candidate.diagnostic.actual_sphere_center_bits[probe][2] =
            g1_ik_runtime_float_bits(
                candidate.sphere_centers[probe].z);
    }

    G1SwingClearanceValidation validation = {};
    const G1ClearanceStatus real_status =
        g1_swing_clearance_validate(
            validation,
            g1_swing_foot_clearance_budget(),
            history,
            field,
            config,
            candidate.sphere_centers,
            false,
            dt,
            error,
            error_capacity);
    if (g1_ik_runtime_clearance_status_is_fatal(real_status) ||
        (real_status != G1ClearanceOk &&
         !g1_ik_runtime_clearance_status_is_finite_rejection(
             real_status))) {
        return false;
    }
    candidate.diagnostic.clearance_status = real_status;
    if (real_status == G1ClearanceOk) {
        candidate.diagnostic.lower_margin_m =
            validation.lower_margin_m;
        candidate.diagnostic.witness_upper_margin_m =
            validation.witness_upper_m;
        candidate.diagnostic.clearance_work = validation.work;
    }

    candidate.diagnostic.clearance_certified =
        candidate.diagnostic.clearance_status == G1ClearanceOk &&
        candidate.diagnostic.lower_margin_m >= 0.0;
    candidate.passes =
        candidate.diagnostic.controller_constraints_passed &&
        candidate.diagnostic.clearance_certified;
    output = candidate;
    return true;
}

#if defined(G1_IK_ENABLE_TEST_SEAMS)

static inline bool g1_ik_stage_swing_candidate_for_test(
    G1SwingCandidateDiagnostic& diagnostic,
    const slice1d<vec3> local_positions,
    const slice1d<quat> baseline_rotations,
    const slice1d<int> bone_parents,
    const G1SwingHistory& history,
    const heightfield& field,
    const G1LegConfig& config,
    const G1FootTarget& target,
    uint32_t candidate_index,
    float dt,
    char* error,
    int error_capacity)
{
    G1IkRuntimeStagedCandidate candidate = {};
    if (!g1_ik_runtime_stage_swing_candidate(
            candidate,
            local_positions,
            baseline_rotations,
            bone_parents,
            history,
            field,
            config,
            target,
            candidate_index,
            dt,
            error,
            error_capacity)) {
        return false;
    }
    diagnostic = candidate.diagnostic;
    return true;
}

#endif

static inline bool g1_ik_runtime_stage_recorded_contact(
    G1IkRuntimeStagedCandidate& output,
    const slice1d<vec3> local_positions,
    const slice1d<quat> baseline_rotations,
    const slice1d<int> bone_parents,
    const G1LegConfig& config,
    const G1FootTarget& target,
    char* error,
    int error_capacity)
{
    G1IkRuntimeStagedCandidate candidate = {};
    std::memcpy(
        candidate.positions,
        local_positions.data,
        sizeof(candidate.positions));
    std::memcpy(
        candidate.rotations,
        baseline_rotations.data,
        sizeof(candidate.rotations));
    const slice1d<vec3> staged_positions(
        G1_BoneCount, candidate.positions);
    const slice1d<quat> staged_rotations(
        G1_BoneCount, candidate.rotations);
    if (!g1_apply_named_physical_sole_ik(
            staged_rotations,
            staged_positions,
            baseline_rotations,
            bone_parents,
            config,
            target.sole_center,
            target.desired_sole_normal,
            candidate.position,
            candidate.orientation,
            error,
            error_capacity)) {
        return false;
    }
    bool baseline_fallback_used = false;
    if (!candidate.position.reachable) {
        vec3 baseline_globals_storage[G1_BoneCount];
        quat baseline_global_rotations_storage[G1_BoneCount];
        const slice1d<vec3> baseline_globals(
            G1_BoneCount, baseline_globals_storage);
        const slice1d<quat> baseline_global_rotations(
            G1_BoneCount, baseline_global_rotations_storage);
        double baseline_residual_precise_m = DBL_MAX;
        float baseline_residual_m = FLT_MAX;
        vec3 baseline_sole_center;
        if (!g1_ik_checked_forward_kinematics(
                baseline_globals,
                baseline_global_rotations,
                staged_positions,
                baseline_rotations,
                bone_parents,
                error,
                error_capacity) ||
            !g1_ik_checked_physical_sole_centroid(
                baseline_sole_center,
                baseline_globals(config.contact),
                baseline_global_rotations(config.contact),
                config) ||
            !ik_checked_distance_precise(
                baseline_residual_precise_m,
                baseline_residual_m,
                baseline_sole_center,
                target.sole_center)) {
            return g1_ik_error(
                error, error_capacity,
                "G1 recorded contact baseline residual became invalid");
        }
        if (g1_ik_contact_residual_is_converged_precise(
                baseline_residual_precise_m)) {
            baseline_fallback_used = true;
            std::memcpy(
                candidate.rotations,
                baseline_rotations.data,
                sizeof(candidate.rotations));
            candidate.position = G1LegSolveResult{};
            if (!g1_apply_named_position_ik(
                    staged_rotations,
                    staged_positions,
                    baseline_rotations,
                    bone_parents,
                    config,
                    baseline_globals(config.ankle),
                    candidate.position,
                    error,
                    error_capacity)) {
                return false;
            }
            if (!g1_ik_leg_result_is_valid(candidate.position) ||
                !candidate.position.reachable ||
                candidate.position.correction_limited ||
                candidate.position.safe_stop_requested ||
                candidate.position.max_correction_radians != 0.0f ||
                !g1_ik_vec3_bits_equal(
                    candidate.position.requested_ankle_target,
                    candidate.position.clamped_ankle_target) ||
                g1_ik_runtime_float_bits(
                    candidate.position.raw_distance_m) !=
                    g1_ik_runtime_float_bits(
                        candidate.position.clamped_distance_m)) {
                return g1_ik_error(
                    error, error_capacity,
                    "G1 recorded contact fallback was incoherent");
            }
            if (!g1_apply_named_foot_orientation(
                    staged_rotations,
                    staged_positions,
                    baseline_rotations,
                    bone_parents,
                    config,
                    target.desired_sole_normal,
                    candidate.orientation,
                    error,
                    error_capacity)) {
                return false;
            }
        }
    }
    vec3 globals_storage[G1_BoneCount];
    quat rotations_storage[G1_BoneCount];
    const slice1d<vec3> globals(
        G1_BoneCount, globals_storage);
    const slice1d<quat> global_rotations(
        G1_BoneCount, rotations_storage);
    vec3 final_sole_center;
    double final_residual_precise_m = DBL_MAX;
    float final_residual_m = FLT_MAX;
    if (!g1_ik_checked_forward_kinematics(
            globals,
            global_rotations,
            staged_positions,
            staged_rotations,
            bone_parents,
            error,
            error_capacity) ||
        !g1_ik_runtime_compute_foot_centers(
            candidate.sphere_centers,
            globals,
            global_rotations,
            config) ||
        !g1_ik_checked_physical_sole_centroid(
            final_sole_center,
            globals(config.contact),
            global_rotations(config.contact),
            config) ||
        !ik_checked_distance_precise(
            final_residual_precise_m,
            final_residual_m,
            final_sole_center,
            target.sole_center)) {
        return false;
    }
    if (baseline_fallback_used) {
        candidate.position.contact_residual_m = final_residual_m;
        candidate.position.iteration_provenance =
            G1LegIterationBaselineFallback1;
        candidate.position.safe_stop_requested =
            !candidate.position.reachable ||
            candidate.position.correction_limited ||
            !g1_ik_contact_residual_is_converged_precise(
                final_residual_precise_m);
        if (!g1_ik_leg_result_is_valid(candidate.position)) {
            return g1_ik_error(
                error, error_capacity,
                "G1 recorded contact fallback final residual was invalid");
        }
    }
    candidate.passes =
        g1_ik_runtime_controller_constraints_pass(
            candidate.position,
            candidate.orientation,
            globals(config.contact),
            global_rotations(config.contact),
            target,
            config);
    output = candidate;
    return true;
}

static inline bool g1_ik_runtime_is_disabled_noop(
    const G1IkFrameTransaction& transaction);

static inline bool g1_ik_frame_stage_foot(
    G1IkFrameTransaction& transaction,
    array1d<vec3>& scratch_positions,
    array1d<quat>& scratch_rotations,
    uint32_t foot_index,
    const slice1d<int> bone_parents,
    const slice1d<bool> contacts,
    const heightfield& field,
    const G1FootprintObservation& footprint,
    bool enabled,
    float dt,
    char* error,
    int error_capacity)
{
    if (error_capacity < 0 ||
        !transaction.initialized ||
        transaction.next_foot >= 2 ||
        foot_index != transaction.next_foot ||
        !g1_dt_is_exact_25_hz(dt) ||
        field.version != 2 ||
        !terrain_heightfield_is_queryable(field) ||
        !g1_ik_runtime_arrays_are_exact(
            scratch_positions, scratch_rotations) ||
        !g1_ik_runtime_pose_shapes_are_exact(
            scratch_positions,
            scratch_rotations,
            bone_parents) ||
        !g1_ik_runtime_pose_values_are_valid(
            scratch_positions,
            scratch_rotations) ||
        !g1_ik_parent_topology_validate(
            bone_parents, error, error_capacity) ||
        contacts.size != 2 || contacts.data == NULL ||
        footprint.feet[0].current_contact != contacts.data[0] ||
        footprint.feet[1].current_contact != contacts.data[1] ||
        !g1_ik_runtime_state_is_valid(
            transaction.candidate_state) ||
        !g1_ik_runtime_iteration_transcript_matches(transaction)) {
        return false;
    }

    const bool disabled_noop =
        g1_ik_runtime_is_disabled_noop(transaction);
    if (!enabled) {
        if (!disabled_noop) return false;
        G1IkFrameTransaction candidate = transaction;
        ++candidate.next_foot;
        transaction = candidate;
        return true;
    }
    if (disabled_noop ||
        contacts.data[0] != transaction.candidate_result
            .feet[0].recorded_contact ||
        contacts.data[1] != transaction.candidate_result
            .feet[1].recorded_contact) {
        return false;
    }

    G1IkFrameTransaction candidate = transaction;
    if (candidate.candidate_result.safe_stop_requested) {
        ++candidate.next_foot;
        transaction = candidate;
        return true;
    }

    vec3 snapshot_positions_storage[G1_BoneCount];
    quat snapshot_rotations_storage[G1_BoneCount];
    std::memcpy(
        snapshot_positions_storage,
        scratch_positions.data,
        sizeof(snapshot_positions_storage));
    std::memcpy(
        snapshot_rotations_storage,
        scratch_rotations.data,
        sizeof(snapshot_rotations_storage));
    const slice1d<vec3> snapshot_positions(
        G1_BoneCount, snapshot_positions_storage);
    const slice1d<quat> snapshot_rotations(
        G1_BoneCount, snapshot_rotations_storage);
    const G1LegConfig config = foot_index == 0
        ? g1_left_leg_config()
        : g1_right_leg_config();
    G1FootFrameResult& foot_result =
        candidate.candidate_result.feet[foot_index];

    if (foot_result.recorded_contact) {
        G1IkRuntimeStagedCandidate planted = {};
        if (!g1_ik_runtime_stage_recorded_contact(
                planted,
                snapshot_positions,
                snapshot_rotations,
                bone_parents,
                config,
                foot_result.target,
                error,
                error_capacity)) {
            return false;
        }
        foot_result.position = planted.position;
        candidate.staged_iteration_provenance[foot_index] =
            planted.position.iteration_provenance;
        foot_result.orientation = planted.orientation;
        if (!planted.passes) {
            g1_ik_runtime_request_safe_stop(
                candidate, G1IkStopTargetUnreachable);
        } else {
            std::memcpy(
                scratch_positions.data,
                planted.positions,
                sizeof(planted.positions));
            std::memcpy(
                scratch_rotations.data,
                planted.rotations,
                sizeof(planted.rotations));
        }
        ++candidate.next_foot;
        transaction = candidate;
        return true;
    }

    G1SwingSelectionDiagnostic& selection =
        foot_result.swing_selection;
    bool selected = false;
    for (uint32_t candidate_index = 0;
         candidate_index < G1SwingLiftCandidateCount;
         ++candidate_index) {
        G1IkRuntimeStagedCandidate staged = {};
        if (!g1_ik_runtime_stage_swing_candidate(
                staged,
                snapshot_positions,
                snapshot_rotations,
                bone_parents,
                candidate.candidate_state.feet[foot_index].swing,
                field,
                config,
                foot_result.target,
                candidate_index,
                dt,
                error,
                error_capacity)) {
            return false;
        }
        ++selection.candidates_evaluated;
        if (staged.diagnostic.clearance_status == G1ClearanceOk &&
            !g1_ik_runtime_work_add(
                selection.total_clearance_work,
                staged.diagnostic.clearance_work)) {
            return g1_ik_runtime_fail(
                error, error_capacity,
                "G1 IK swing aggregate work overflowed");
        }
        if (staged.passes) {
            selection.selected_index = candidate_index;
            selection.selected = staged.diagnostic;
            foot_result.position = staged.position;
            candidate.staged_iteration_provenance[foot_index] =
                staged.position.iteration_provenance;
            foot_result.orientation = staged.orientation;
            std::memcpy(
                scratch_positions.data,
                staged.positions,
                sizeof(staged.positions));
            std::memcpy(
                scratch_rotations.data,
                staged.rotations,
                sizeof(staged.rotations));
            selected = true;
            break;
        }
        if (staged.diagnostic.clearance_status != G1ClearanceOk &&
            !g1_ik_runtime_clearance_status_is_finite_rejection(
                staged.diagnostic.clearance_status)) {
            return g1_ik_runtime_fail(
                error, error_capacity,
                "G1 IK swing candidate returned an invalid status");
        }
    }

    if (!selected) {
        g1_ik_runtime_request_safe_stop(
            candidate, G1IkStopNoSwingCandidate);
    }
    ++candidate.next_foot;
    transaction = candidate;
    return true;
}

static inline bool g1_ik_runtime_stop_reason_is_valid(
    G1IkStopReason reason)
{
    return reason >= G1IkStopNone &&
           reason <= G1IkStopPoseClearanceRejected;
}

static inline bool g1_ik_runtime_is_disabled_noop(
    const G1IkFrameTransaction& transaction)
{
    const auto float_is_positive_zero = [](float value) {
        return g1_ik_runtime_float_bits(value) == 0U;
    };
    const auto double_is_positive_zero = [](double value) {
        uint64_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        return bits == 0U;
    };
    const auto vec3_is_positive_zero =
        [&float_is_positive_zero](vec3 value) {
            return float_is_positive_zero(value.x) &&
                   float_is_positive_zero(value.y) &&
                   float_is_positive_zero(value.z);
        };
    const auto work_is_zero = [](const G1ClearanceWork& work) {
        return work.point_queries == 0U &&
               work.cells_visited == 0U &&
               work.primitive_triangle_pairs == 0U &&
               work.face_patches == 0U &&
               work.candidate_tests == 0U &&
               work.subdivision_nodes == 0U;
    };
    if (!g1_ik_runtime_iteration_transcript_matches(transaction) ||
        transaction.candidate_result.applied ||
        transaction.candidate_result.safe_stop_requested ||
        transaction.candidate_result.stop_reason != G1IkStopNone ||
        !float_is_positive_zero(
            transaction.candidate_result.max_correction_radians)) {
        return false;
    }
    for (int foot = 0; foot < 2; ++foot) {
        const G1FootFrameResult& result =
            transaction.candidate_result.feet[foot];
        const G1FootTarget& target = result.target;
        const G1SwingSelectionDiagnostic& selection =
            result.swing_selection;
        const G1SwingCandidateDiagnostic& selected =
            selection.selected;
        const G1SwingClearanceValidation& defensive =
            result.defensive_swing;
        const G1LegSolveResult& position = result.position;
        const G1FootOrientationResult& orientation =
            result.orientation;
        if (result.recorded_contact ||
            target.locked ||
            target.position_active ||
            target.releasing ||
            target.drift_limit_exceeded ||
            !vec3_is_positive_zero(target.surface.point) ||
            !vec3_is_positive_zero(target.surface.normal) ||
            !vec3_is_positive_zero(target.desired_sole_normal) ||
            !vec3_is_positive_zero(target.sole_center) ||
            !float_is_positive_zero(target.horizontal_drift_m) ||
            selection.candidates_evaluated != 0U ||
            result.swing_selection.selected_index != G1SwingNoCandidate ||
            selected.candidate_index != G1SwingNoCandidate ||
            selected.lift_bits != 0U ||
            selected.materialized_command_y_bits != 0U ||
            selected.clearance_status != G1ClearanceInvalidInput ||
            selected.controller_constraints_passed ||
            selected.clearance_certified ||
            !double_is_positive_zero(selected.lower_margin_m) ||
            !double_is_positive_zero(
                selected.witness_upper_margin_m) ||
            !work_is_zero(selected.clearance_work) ||
            !work_is_zero(selection.total_clearance_work) ||
            !double_is_positive_zero(defensive.lower_margin_m) ||
            !double_is_positive_zero(defensive.witness_upper_m) ||
            defensive.sweep_evaluated ||
            !work_is_zero(defensive.work) ||
            position.applied ||
            position.reachable ||
            position.correction_limited ||
            position.safe_stop_requested ||
            position.iterations != 0 ||
            position.iteration_provenance !=
                G1LegIterationNone ||
            !vec3_is_positive_zero(position.requested_ankle_target) ||
            !vec3_is_positive_zero(position.clamped_ankle_target) ||
            !vec3_is_positive_zero(position.hinge_axis_world) ||
            !vec3_is_positive_zero(position.bend_direction) ||
            position.bend_used_current_projection ||
            position.bend_used_hinge_fallback ||
            position.bend_used_safe_perpendicular ||
            position.bend_sign_flipped ||
            !float_is_positive_zero(position.raw_distance_m) ||
            !float_is_positive_zero(position.clamped_distance_m) ||
            !float_is_positive_zero(
                position.max_correction_radians) ||
            g1_ik_runtime_float_bits(position.contact_residual_m) !=
                g1_ik_runtime_float_bits(
                    std::numeric_limits<float>::max()) ||
            orientation.applied ||
            orientation.correction_limited ||
            orientation.safe_stop_requested ||
            g1_ik_runtime_float_bits(
                orientation.target_global_rotation.w) !=
                g1_ik_runtime_float_bits(1.0f) ||
            !float_is_positive_zero(
                orientation.target_global_rotation.x) ||
            !float_is_positive_zero(
                orientation.target_global_rotation.y) ||
            !float_is_positive_zero(
                orientation.target_global_rotation.z) ||
            !float_is_positive_zero(
                orientation.requested_correction_radians) ||
            !float_is_positive_zero(
                orientation.correction_radians)) {
            return false;
        }
        for (int probe = 0; probe < 4; ++probe) {
            for (int axis = 0; axis < 3; ++axis) {
                if (selected.actual_sphere_center_bits[probe][axis] !=
                    0U) {
                    return false;
                }
            }
        }
    }
    return true;
}

static inline bool g1_ik_frame_rejection_snapshot(
    G1IkFrameResult& output,
    const G1IkFrameTransaction& transaction,
    G1IkRejectionCheckpoint expected_checkpoint,
    char* error,
    int error_capacity)
{
    if (error_capacity < 0) return false;

    const std::size_t error_bytes =
        error != NULL && error_capacity > 0
            ? static_cast<std::size_t>(error_capacity)
            : 0U;
    if (g1_ik_memory_ranges_overlap(
            error, error_bytes, &output, sizeof(output)) ||
        g1_ik_memory_ranges_overlap(
            error, error_bytes,
            &transaction, sizeof(transaction))) {
        return false;
    }
    if (g1_ik_memory_ranges_overlap(
            &output, sizeof(output),
            &transaction, sizeof(transaction))) {
        return g1_ik_runtime_fail(
            error, error_capacity,
            "G1 IK rejection snapshot output aliases its transaction");
    }

    static_assert(
        sizeof(G1IkRejectionCheckpoint) == sizeof(int),
        "G1 IK rejection checkpoint uses an exact int representation");
    int checkpoint = -1;
    std::memcpy(
        &checkpoint, &expected_checkpoint,
        sizeof(expected_checkpoint));
    if (checkpoint < G1IkRejectionAfterBegin ||
        checkpoint > G1IkRejectionAfterFoot1) {
        return g1_ik_runtime_fail(
            error, error_capacity,
            "G1 IK rejection snapshot checkpoint is invalid");
    }

    const auto double_bits = [](double value) {
        uint64_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        return bits;
    };
    const auto float_is_positive_zero = [](float value) {
        return g1_ik_runtime_float_bits(value) == 0U;
    };
    const auto double_is_positive_zero = [&double_bits](double value) {
        return double_bits(value) == 0U;
    };
    const auto vec3_is_positive_zero =
        [&float_is_positive_zero](vec3 value) {
            return float_is_positive_zero(value.x) &&
                   float_is_positive_zero(value.y) &&
                   float_is_positive_zero(value.z);
        };
    const auto work_is_zero = [](const G1ClearanceWork& work) {
        return work.point_queries == 0U &&
               work.cells_visited == 0U &&
               work.primitive_triangle_pairs == 0U &&
               work.face_patches == 0U &&
               work.candidate_tests == 0U &&
               work.subdivision_nodes == 0U;
    };
    const auto work_is_at_least = [](
        const G1ClearanceWork& whole,
        const G1ClearanceWork& part) {
        return whole.point_queries >= part.point_queries &&
               whole.cells_visited >= part.cells_visited &&
               whole.primitive_triangle_pairs >=
                   part.primitive_triangle_pairs &&
               whole.face_patches >= part.face_patches &&
               whole.candidate_tests >= part.candidate_tests &&
               whole.subdivision_nodes >=
                   part.subdivision_nodes;
    };
    const G1ClearanceBudget swing_budget =
        g1_swing_foot_clearance_budget();
    const auto work_is_within = [&swing_budget](
        const G1ClearanceWork& work,
        uint32_t multiplier) {
        return static_cast<uint64_t>(work.point_queries) <=
                   static_cast<uint64_t>(
                       swing_budget.maximum_point_queries) * multiplier &&
               static_cast<uint64_t>(work.cells_visited) <=
                   static_cast<uint64_t>(
                       swing_budget.maximum_cells) * multiplier &&
               static_cast<uint64_t>(
                   work.primitive_triangle_pairs) <=
                   static_cast<uint64_t>(
                       swing_budget.maximum_primitive_triangle_pairs) *
                       multiplier &&
               static_cast<uint64_t>(work.face_patches) <=
                   static_cast<uint64_t>(
                       swing_budget.maximum_face_patches) * multiplier &&
               static_cast<uint64_t>(work.candidate_tests) <=
                   static_cast<uint64_t>(
                       swing_budget.maximum_candidate_tests) * multiplier &&
               static_cast<uint64_t>(work.subdivision_nodes) <=
                   static_cast<uint64_t>(
                       swing_budget.maximum_subdivision_nodes) * multiplier;
    };
    const auto candidate_is_canonical = [
        &double_is_positive_zero, &work_is_zero](
        const G1SwingCandidateDiagnostic& selected) {
        if (selected.candidate_index != G1SwingNoCandidate ||
            selected.lift_bits != 0U ||
            selected.materialized_command_y_bits != 0U ||
            selected.clearance_status != G1ClearanceInvalidInput ||
            selected.controller_constraints_passed ||
            selected.clearance_certified ||
            !double_is_positive_zero(selected.lower_margin_m) ||
            !double_is_positive_zero(
                selected.witness_upper_margin_m) ||
            !work_is_zero(selected.clearance_work)) {
            return false;
        }
        for (int probe = 0; probe < 4; ++probe) {
            for (int axis = 0; axis < 3; ++axis) {
                if (selected.actual_sphere_center_bits[probe][axis] !=
                    0U) {
                    return false;
                }
            }
        }
        return true;
    };
    const auto selection_is_canonical = [
        &candidate_is_canonical, &work_is_zero](
        const G1SwingSelectionDiagnostic& selection) {
        return selection.candidates_evaluated == 0U &&
               selection.selected_index == G1SwingNoCandidate &&
               candidate_is_canonical(selection.selected) &&
               work_is_zero(selection.total_clearance_work);
    };
    const auto defensive_is_canonical = [
        &double_is_positive_zero, &work_is_zero](
        const G1SwingClearanceValidation& defensive) {
        return double_is_positive_zero(defensive.lower_margin_m) &&
               double_is_positive_zero(defensive.witness_upper_m) &&
               !defensive.sweep_evaluated &&
               work_is_zero(defensive.work);
    };
    const auto position_is_canonical = [
        &float_is_positive_zero, &vec3_is_positive_zero](
        const G1LegSolveResult& position) {
        return !position.applied && !position.reachable &&
               !position.correction_limited &&
               !position.safe_stop_requested &&
               position.iterations == 0 &&
               position.iteration_provenance ==
                   G1LegIterationNone &&
               vec3_is_positive_zero(
                   position.requested_ankle_target) &&
               vec3_is_positive_zero(
                   position.clamped_ankle_target) &&
               vec3_is_positive_zero(position.hinge_axis_world) &&
               vec3_is_positive_zero(position.bend_direction) &&
               !position.bend_used_current_projection &&
               !position.bend_used_hinge_fallback &&
               !position.bend_used_safe_perpendicular &&
               !position.bend_sign_flipped &&
               float_is_positive_zero(position.raw_distance_m) &&
               float_is_positive_zero(position.clamped_distance_m) &&
               float_is_positive_zero(
                   position.max_correction_radians) &&
               g1_ik_runtime_float_bits(
                   position.contact_residual_m) ==
                   g1_ik_runtime_float_bits(
                       std::numeric_limits<float>::max());
    };
    const auto orientation_is_canonical = [
        &float_is_positive_zero](
        const G1FootOrientationResult& orientation) {
        return !orientation.applied &&
               !orientation.correction_limited &&
               !orientation.safe_stop_requested &&
               g1_ik_runtime_float_bits(
                   orientation.target_global_rotation.w) ==
                   g1_ik_runtime_float_bits(1.0f) &&
               float_is_positive_zero(
                   orientation.target_global_rotation.x) &&
               float_is_positive_zero(
                   orientation.target_global_rotation.y) &&
               float_is_positive_zero(
                   orientation.target_global_rotation.z) &&
               float_is_positive_zero(
                   orientation.requested_correction_radians) &&
               float_is_positive_zero(
                   orientation.correction_radians);
    };
    const auto foot_stage_is_canonical = [
        &selection_is_canonical,
        &position_is_canonical,
        &orientation_is_canonical](
        const G1FootFrameResult& foot) {
        return selection_is_canonical(foot.swing_selection) &&
               position_is_canonical(foot.position) &&
               orientation_is_canonical(foot.orientation);
    };
    const auto position_is_complete = [](
        const G1LegSolveResult& position,
        const G1LegConfig& config) {
        const int primary_bend_provenance =
            (position.bend_used_current_projection ? 1 : 0) +
            (position.bend_used_hinge_fallback ? 1 : 0);
        const bool residual_converged =
            g1_ik_contact_residual_is_converged(
                position.contact_residual_m);
        const bool residual_at_threshold =
            g1_ik_runtime_float_bits(
                position.contact_residual_m) ==
            g1_ik_runtime_float_bits(0.005f);
        const bool definite_constraints_failed =
            !position.reachable ||
            position.correction_limited ||
            !residual_converged;
        const bool safe_stop_is_coherent =
            definite_constraints_failed
                ? position.safe_stop_requested
                : (residual_at_threshold ||
                   !position.safe_stop_requested);
        const bool reachable_projection_matches =
            !position.reachable ||
            (g1_ik_vec3_bits_equal(
                 position.requested_ankle_target,
                 position.clamped_ankle_target) &&
             g1_ik_runtime_float_bits(
                 position.raw_distance_m) ==
                 g1_ik_runtime_float_bits(
                     position.clamped_distance_m));
        return g1_ik_leg_result_is_valid(position) &&
               g1_ik_contact_iterations_have_valid_provenance(
                   position) &&
               position.max_correction_radians <=
                   config.max_correction_radians &&
               primary_bend_provenance == 1 &&
               (!position.bend_used_safe_perpendicular ||
                position.bend_used_hinge_fallback) &&
               (!position.bend_sign_flipped ||
                position.bend_used_hinge_fallback) &&
               (!position.correction_limited ||
                position.max_correction_radians > 0.0f) &&
               safe_stop_is_coherent &&
               reachable_projection_matches;
    };
    const auto orientation_is_complete = [](
        const G1FootOrientationResult& orientation,
        const G1FootTarget& target,
        const G1LegConfig& config) {
        vec3 aligned_sole_normal;
        vec3 normalized_target_normal;
        double normal_alignment = 0.0;
        const bool target_alignment_is_valid =
            ik_checked_normalize(
                normalized_target_normal,
                target.desired_sole_normal) &&
            ik_checked_quat_rotate(
                aligned_sole_normal,
                orientation.target_global_rotation,
                config.sole_normal_local) &&
            ik_checked_dot(
                normal_alignment,
                aligned_sole_normal,
                normalized_target_normal) &&
            normal_alignment >= 0.99999;
        return orientation.applied &&
               ik_quat_is_unit(
                   orientation.target_global_rotation) &&
               g1_ik_float_is_runtime_value(
                   orientation.requested_correction_radians) &&
               orientation.requested_correction_radians >= 0.0f &&
               g1_ik_float_is_runtime_value(
                   orientation.correction_radians) &&
               orientation.correction_radians >= 0.0f &&
               orientation.correction_radians <=
                   orientation.requested_correction_radians &&
               orientation.correction_radians <=
                   config.max_correction_radians &&
               orientation.safe_stop_requested ==
                   orientation.correction_limited &&
               (!orientation.correction_limited ||
                (orientation.requested_correction_radians >=
                     config.max_correction_radians &&
                 orientation.correction_radians > 0.0f)) &&
               (orientation.correction_limited ||
                g1_ik_runtime_float_bits(
                    orientation.requested_correction_radians) ==
                    g1_ik_runtime_float_bits(
                        orientation.correction_radians)) &&
               target_alignment_is_valid;
    };
    const auto staged_constraints_pass = [](
        const G1LegSolveResult& position,
        const G1FootOrientationResult& orientation) {
        return position.reachable &&
               !position.correction_limited &&
               !position.safe_stop_requested &&
               g1_ik_contact_residual_is_converged(
                   position.contact_residual_m) &&
               !orientation.correction_limited &&
               !orientation.safe_stop_requested;
    };
    const auto float_word_is_runtime_value = [](uint32_t bits) {
        float value = 0.0f;
        std::memcpy(&value, &bits, sizeof(value));
        return g1_ik_float_is_runtime_value(value);
    };
    const auto selected_swing_is_valid = [
        &float_word_is_runtime_value,
        &work_is_at_least,
        &work_is_within](
        const G1FootFrameResult& foot) {
        const G1SwingSelectionDiagnostic& selection =
            foot.swing_selection;
        if (selection.candidates_evaluated == 0U ||
            selection.candidates_evaluated >
                G1SwingLiftCandidateCount ||
            selection.selected_index >=
                selection.candidates_evaluated ||
            selection.candidates_evaluated !=
                selection.selected_index + 1U ||
            selection.selected_index >=
                G1SwingLiftCandidateCount) {
            return false;
        }
        const G1SwingCandidateDiagnostic& selected =
            selection.selected;
        if (selected.candidate_index !=
                selection.selected_index ||
            selected.lift_bits !=
                G1SwingLiftCandidateBits[
                    selection.selected_index] ||
            !float_word_is_runtime_value(
                selected.materialized_command_y_bits) ||
            selected.clearance_status != G1ClearanceOk ||
            !selected.controller_constraints_passed ||
            !selected.clearance_certified ||
            !terrain_double_is_finite(selected.lower_margin_m) ||
            selected.lower_margin_m < 0.0 ||
            !terrain_double_is_finite(
                selected.witness_upper_margin_m) ||
            selected.witness_upper_margin_m <
                selected.lower_margin_m) {
            return false;
        }
        const volatile double certificate_width =
            selected.witness_upper_margin_m -
            selected.lower_margin_m;
        if (!terrain_double_is_finite(certificate_width) ||
            certificate_width < 0.0 ||
            certificate_width >
                G1ClearanceMaximumCertificateWidthM ||
            !work_is_within(selected.clearance_work, 1U) ||
            !work_is_within(
                selection.total_clearance_work,
                selection.candidates_evaluated) ||
            !work_is_at_least(
                selection.total_clearance_work,
                selected.clearance_work)) {
            return false;
        }
        float lift = 0.0f;
        std::memcpy(
            &lift, &selected.lift_bits, sizeof(lift));
        float expected_y = 0.0f;
        if (g1_apply_swing_lift_y(
                expected_y, foot.target.sole_center.y, lift,
                NULL, 0) != G1ClearanceOk ||
            g1_ik_runtime_float_bits(expected_y) !=
                selected.materialized_command_y_bits) {
            return false;
        }
        for (int probe = 0; probe < 4; ++probe) {
            for (int axis = 0; axis < 3; ++axis) {
                if (!float_word_is_runtime_value(
                        selected.actual_sphere_center_bits[
                            probe][axis])) {
                    return false;
                }
            }
        }
        return true;
    };
    const auto no_swing_is_valid = [
        &candidate_is_canonical,
        &work_is_within,
        &position_is_canonical,
        &orientation_is_canonical](
        const G1FootFrameResult& foot) {
        const G1SwingSelectionDiagnostic& selection =
            foot.swing_selection;
        return !foot.recorded_contact &&
               selection.candidates_evaluated ==
                   G1SwingLiftCandidateCount &&
               selection.selected_index == G1SwingNoCandidate &&
               candidate_is_canonical(selection.selected) &&
               work_is_within(
                   selection.total_clearance_work,
                   G1SwingLiftCandidateCount) &&
               position_is_canonical(foot.position) &&
               orientation_is_canonical(foot.orientation);
    };
    const auto successful_foot_is_valid = [
        &selection_is_canonical,
        &position_is_complete,
        &orientation_is_complete,
        &staged_constraints_pass,
        &selected_swing_is_valid](
        const G1FootFrameResult& foot,
        const G1LegConfig& config) {
        if (!position_is_complete(foot.position, config) ||
            !orientation_is_complete(
                foot.orientation, foot.target, config) ||
            !staged_constraints_pass(
                foot.position, foot.orientation)) {
            return false;
        }
        return foot.recorded_contact
            ? selection_is_canonical(foot.swing_selection)
            : selected_swing_is_valid(foot);
    };
    const auto rejecting_foot_is_valid = [
        &selection_is_canonical,
        &position_is_complete,
        &orientation_is_complete,
        &staged_constraints_pass,
        &no_swing_is_valid](
        const G1FootFrameResult& foot,
        const G1LegConfig& config,
        G1IkStopReason reason) {
        if (reason == G1IkStopTargetUnreachable) {
            return foot.recorded_contact &&
                   selection_is_canonical(
                       foot.swing_selection) &&
                   position_is_complete(foot.position, config) &&
                   orientation_is_complete(
                       foot.orientation, foot.target, config) &&
                   !staged_constraints_pass(
                       foot.position, foot.orientation);
        }
        return reason == G1IkStopNoSwingCandidate &&
               no_swing_is_valid(foot);
    };

    int reason = -1;
    static_assert(
        sizeof(G1IkStopReason) == sizeof(reason),
        "G1 IK stop reason uses an exact int representation");
    std::memcpy(
        &reason,
        &transaction.candidate_result.stop_reason,
        sizeof(reason));
    if (!transaction.initialized ||
        !g1_ik_runtime_state_is_valid(
            transaction.candidate_state) ||
        !g1_ik_runtime_iteration_transcript_matches(transaction) ||
        transaction.candidate_result.applied ||
        !transaction.candidate_result.safe_stop_requested ||
        reason <= G1IkStopNone ||
        reason > G1IkStopPoseClearanceRejected ||
        !float_is_positive_zero(
            transaction.candidate_result
                .max_correction_radians)) {
        return g1_ik_runtime_fail(
            error, error_capacity,
            "G1 IK rejection snapshot has invalid common fields");
    }

    const G1LegConfig configs[2] = {
        g1_left_leg_config(), g1_right_leg_config()
    };
    bool noncontact_base_target[2] = {false, false};
    for (int foot_index = 0; foot_index < 2; ++foot_index) {
        const G1FootIkState& foot_state =
            transaction.candidate_state.feet[foot_index];
        const G1FootFrameResult& foot =
            transaction.candidate_result.feet[foot_index];
        bool recorded_contact_provenance_is_valid = true;
        bool noncontact_provenance_is_valid = true;
        if (foot.recorded_contact) {
            float expected_horizontal_drift = 0.0f;
            bool expected_drift_limit_exceeded = false;
            recorded_contact_provenance_is_valid =
                g1_ik_vec3_bits_equal(
                    foot.target.sole_center,
                    foot_state.lock.lock_point) &&
                g1_ik_vec3_bits_equal(
                    foot.target.surface.point,
                    foot_state.lock.lock_point) &&
                g1_ik_vec3_bits_equal(
                    foot.target.desired_sole_normal,
                    foot.target.surface.normal) &&
                g1_foot_horizontal_drift(
                    expected_horizontal_drift,
                    expected_drift_limit_exceeded,
                    foot_state.lock.previous_input,
                    foot_state.lock.lock_point) &&
                g1_ik_runtime_float_bits(
                    foot.target.horizontal_drift_m) ==
                    g1_ik_runtime_float_bits(
                        expected_horizontal_drift) &&
                foot.target.drift_limit_exceeded ==
                    expected_drift_limit_exceeded;
        } else {
            noncontact_base_target[foot_index] =
                g1_ik_vec3_bits_equal(
                    foot.target.sole_center,
                    foot_state.lock.output_position) &&
                g1_ik_runtime_float_bits(
                    foot.target.surface.point.x) ==
                    g1_ik_runtime_float_bits(
                        foot_state.lock.previous_input.x) &&
                g1_ik_runtime_float_bits(
                    foot.target.surface.point.z) ==
                    g1_ik_runtime_float_bits(
                    foot_state.lock.previous_input.z);
            noncontact_provenance_is_valid =
                g1_ik_runtime_float_bits(
                    foot.target.horizontal_drift_m) == 0U &&
                !foot.target.drift_limit_exceeded &&
                g1_ik_vec3_bits_equal(
                    foot.target.desired_sole_normal,
                    foot_state.baseline_sole_normal) &&
                noncontact_base_target[foot_index];
        }
        if (!g1_foot_target_is_valid(foot.target) ||
            !g1_ik_surface_target_is_valid(
                foot.target.surface) ||
            foot.recorded_contact != foot_state.lock.contact ||
            foot.target.locked != foot_state.lock.locked ||
            foot.target.position_active !=
                foot_state.lock.position_active ||
            foot.target.releasing != foot_state.lock.releasing ||
            !recorded_contact_provenance_is_valid ||
            !noncontact_provenance_is_valid ||
            !defensive_is_canonical(foot.defensive_swing)) {
            return g1_ik_runtime_fail(
                error, error_capacity,
                "G1 IK rejection snapshot has invalid foot provenance");
        }
    }

    const G1IkFrameResult& result =
        transaction.candidate_result;
    bool checkpoint_valid = false;
    if (checkpoint == G1IkRejectionAfterBegin) {
        const bool begin_target_forms_are_valid =
            result.stop_reason == G1IkStopFootprintBlocked
                ? ((result.feet[0].recorded_contact ||
                    noncontact_base_target[0]) &&
                   (result.feet[1].recorded_contact ||
                    noncontact_base_target[1]))
                : ((!result.feet[0].recorded_contact ||
                    !result.feet[1].recorded_contact) &&
                   (!result.feet[1].recorded_contact
                        ? noncontact_base_target[1]
                        : (!result.feet[0].recorded_contact &&
                           noncontact_base_target[0])));
        checkpoint_valid =
            transaction.next_foot == 0U &&
            (result.stop_reason == G1IkStopFootprintBlocked ||
             result.stop_reason ==
                 G1IkStopLandingPatchUnavailable) &&
            begin_target_forms_are_valid &&
            foot_stage_is_canonical(result.feet[0]) &&
            foot_stage_is_canonical(result.feet[1]);
    } else if (checkpoint == G1IkRejectionAfterFoot0) {
        checkpoint_valid =
            transaction.next_foot == 1U &&
            (result.stop_reason == G1IkStopTargetUnreachable ||
             result.stop_reason == G1IkStopNoSwingCandidate) &&
            rejecting_foot_is_valid(
                result.feet[0], configs[0], result.stop_reason) &&
            foot_stage_is_canonical(result.feet[1]);
    } else {
        checkpoint_valid =
            transaction.next_foot == 2U &&
            (result.stop_reason == G1IkStopTargetUnreachable ||
             result.stop_reason == G1IkStopNoSwingCandidate) &&
            successful_foot_is_valid(
                result.feet[0], configs[0]) &&
            rejecting_foot_is_valid(
                result.feet[1], configs[1], result.stop_reason);
    }
    if (!checkpoint_valid) {
        return g1_ik_runtime_fail(
            error, error_capacity,
            "G1 IK rejection snapshot does not match its checkpoint");
    }

    output = result;
    return true;
}

static inline bool g1_ik_frame_finish(
    G1IkState& output_state,
    G1IkFrameResult& output_result,
    G1IkFrameTransaction& transaction,
    array1d<vec3>& scratch_positions,
    array1d<quat>& scratch_rotations,
    const slice1d<int> bone_parents,
    const heightfield& field,
    float dt,
    char* error,
    int error_capacity)
{
    if (error_capacity < 0 ||
        !transaction.initialized ||
        transaction.next_foot != 2 ||
        !g1_ik_runtime_state_is_valid(
            transaction.candidate_state) ||
        !g1_ik_runtime_iteration_transcript_matches(transaction) ||
        !g1_ik_runtime_stop_reason_is_valid(
            transaction.candidate_result.stop_reason) ||
        !g1_dt_is_exact_25_hz(dt) ||
        field.version != 2 ||
        !terrain_heightfield_is_queryable(field) ||
        !g1_ik_runtime_arrays_are_exact(
            scratch_positions, scratch_rotations) ||
        !g1_ik_runtime_pose_shapes_are_exact(
            scratch_positions, scratch_rotations, bone_parents) ||
        !g1_ik_runtime_pose_values_are_valid(
            scratch_positions, scratch_rotations) ||
        !g1_ik_parent_topology_validate(
            bone_parents, error, error_capacity)) {
        return false;
    }

    G1IkFrameTransaction candidate = transaction;
    if (candidate.candidate_result.safe_stop_requested ||
        g1_ik_runtime_is_disabled_noop(candidate)) {
        output_result = candidate.candidate_result;
        transaction = candidate;
        return true;
    }

    vec3 global_positions_storage[G1_BoneCount];
    quat global_rotations_storage[G1_BoneCount];
    const slice1d<vec3> global_positions(
        G1_BoneCount, global_positions_storage);
    const slice1d<quat> global_rotations(
        G1_BoneCount, global_rotations_storage);
    if (!g1_ik_checked_forward_kinematics(
            global_positions,
            global_rotations,
            scratch_positions,
            scratch_rotations,
            bone_parents,
            error,
            error_capacity)) {
        return false;
    }

    const G1LegConfig configs[2] = {
        g1_left_leg_config(), g1_right_leg_config()
    };
    vec3 final_centers[2][4] = {};
    for (int foot_index = 0; foot_index < 2; ++foot_index) {
        if (!g1_ik_runtime_compute_foot_centers(
                final_centers[foot_index],
                global_positions,
                global_rotations,
                configs[foot_index])) {
            return g1_ik_runtime_fail(
                error, error_capacity,
                "G1 IK final FK foot centers are invalid");
        }
        const G1SwingSelectionDiagnostic& selection =
            candidate.candidate_result.feet[foot_index]
                .swing_selection;
        G1FootTarget final_target =
            candidate.candidate_result.feet[foot_index].target;
        if (!candidate.candidate_result.feet[foot_index]
                 .recorded_contact) {
            if (selection.selected_index == G1SwingNoCandidate ||
                selection.selected.candidate_index !=
                    selection.selected_index) {
                return g1_ik_runtime_fail(
                    error, error_capacity,
                    "G1 IK final swing selection is incomplete");
            }
            float lift = 0.0f;
            std::memcpy(
                &lift,
                &selection.selected.lift_bits,
                sizeof(lift));
            if (g1_apply_swing_lift_y(
                    final_target.sole_center.y,
                    final_target.sole_center.y,
                    lift,
                    error,
                    error_capacity) != G1ClearanceOk) {
                return false;
            }
            for (int probe = 0; probe < 4; ++probe) {
                const uint32_t bits[3] = {
                    g1_ik_runtime_float_bits(
                        final_centers[foot_index][probe].x),
                    g1_ik_runtime_float_bits(
                        final_centers[foot_index][probe].y),
                    g1_ik_runtime_float_bits(
                        final_centers[foot_index][probe].z)
                };
                for (int axis = 0; axis < 3; ++axis) {
                    if (bits[axis] != selection.selected
                            .actual_sphere_center_bits[probe][axis]) {
                        return g1_ik_runtime_fail(
                            error, error_capacity,
                            "G1 IK final FK changed a selected endpoint");
                    }
                }
            }
        }
        if (!g1_ik_runtime_controller_constraints_pass(
                candidate.candidate_result.feet[foot_index].position,
                candidate.candidate_result.feet[foot_index].orientation,
                global_positions(configs[foot_index].contact),
                global_rotations(configs[foot_index].contact),
                final_target,
                configs[foot_index])) {
            g1_ik_runtime_request_safe_stop(
                candidate, G1IkStopTargetUnreachable);
            output_result = candidate.candidate_result;
            transaction = candidate;
            return true;
        }

        G1SwingClearanceValidation defensive = {};
        const G1ClearanceStatus status =
            g1_swing_clearance_validate(
                defensive,
                g1_swing_foot_clearance_budget(),
                candidate.candidate_state.feet[foot_index].swing,
                field,
                configs[foot_index],
                final_centers[foot_index],
                candidate.candidate_result.feet[foot_index]
                    .recorded_contact,
                dt,
                error,
                error_capacity);
        if (g1_ik_runtime_clearance_status_is_fatal(status) ||
            (status != G1ClearanceOk &&
             !g1_ik_runtime_clearance_status_is_finite_rejection(
                 status))) {
            return false;
        }
        if (status != G1ClearanceOk ||
            (!candidate.candidate_result.feet[foot_index]
                  .recorded_contact &&
             defensive.lower_margin_m < 0.0)) {
            g1_ik_runtime_request_safe_stop(
                candidate, G1IkStopPoseClearanceRejected);
            output_result = candidate.candidate_result;
            transaction = candidate;
            return true;
        }
        candidate.candidate_result.feet[foot_index]
            .defensive_swing = defensive;
    }

    G1IkState state_candidate = candidate.candidate_state;
    for (int foot_index = 0; foot_index < 2; ++foot_index) {
        if (!g1_swing_history_commit(
                state_candidate.feet[foot_index].swing,
                final_centers[foot_index],
                error,
                error_capacity)) {
            return false;
        }
    }
    if (!g1_ik_runtime_state_is_valid(state_candidate)) {
        return g1_ik_runtime_fail(
            error, error_capacity,
            "G1 IK finish produced invalid accepted state");
    }

    float maximum_correction = 0.0f;
    for (int foot_index = 0; foot_index < 2; ++foot_index) {
        const G1FootFrameResult& foot_result =
            candidate.candidate_result.feet[foot_index];
        if (foot_result.position.max_correction_radians >
            maximum_correction) {
            maximum_correction =
                foot_result.position.max_correction_radians;
        }
        if (foot_result.orientation.correction_radians >
            maximum_correction) {
            maximum_correction =
                foot_result.orientation.correction_radians;
        }
    }
    if (!g1_ik_float_is_runtime_value(maximum_correction)) {
        return g1_ik_runtime_fail(
            error, error_capacity,
            "G1 IK finish correction diagnostic is invalid");
    }
    candidate.candidate_state = state_candidate;
    candidate.candidate_result.applied = true;
    candidate.candidate_result.safe_stop_requested = false;
    candidate.candidate_result.stop_reason = G1IkStopNone;
    candidate.candidate_result.max_correction_radians =
        maximum_correction;

    output_state = candidate.candidate_state;
    output_result = candidate.candidate_result;
    transaction = candidate;
    return true;
}

static inline bool g1_ik_frame_evaluate(
    array1d<vec3>& output_positions,
    array1d<quat>& output_rotations,
    G1IkState& state,
    const slice1d<vec3> baseline_positions,
    const slice1d<quat> baseline_rotations,
    const slice1d<int> bone_parents,
    const slice1d<bool> contacts,
    const heightfield& field,
    const G1FootprintObservation& footprint,
    bool enabled,
    float dt,
    G1IkFrameResult& result,
    char* error,
    int error_capacity)
{
    if (error_capacity < 0 ||
        !g1_ik_runtime_arrays_are_exact(
            output_positions, output_rotations) ||
        !g1_ik_runtime_pose_shapes_are_exact(
            baseline_positions, baseline_rotations, bone_parents) ||
        !g1_ik_runtime_pose_ranges_are_disjoint(
            output_positions,
            output_rotations,
            baseline_positions,
            baseline_rotations,
            bone_parents)) {
        return false;
    }

    array1d<vec3> scratch_positions(G1_BoneCount);
    array1d<quat> scratch_rotations(G1_BoneCount);
    G1IkFrameTransaction transaction = {};
    if (!g1_ik_frame_begin(
            transaction,
            scratch_positions,
            scratch_rotations,
            state,
            baseline_positions,
            baseline_rotations,
            bone_parents,
            contacts,
            field,
            footprint,
            enabled,
            dt,
            error,
            error_capacity) ||
        !g1_ik_frame_stage_foot(
            transaction,
            scratch_positions,
            scratch_rotations,
            0,
            bone_parents,
            contacts,
            field,
            footprint,
            enabled,
            dt,
            error,
            error_capacity) ||
        !g1_ik_frame_stage_foot(
            transaction,
            scratch_positions,
            scratch_rotations,
            1,
            bone_parents,
            contacts,
            field,
            footprint,
            enabled,
            dt,
            error,
            error_capacity)) {
        return false;
    }

    G1IkState state_candidate = state;
    G1IkFrameResult result_candidate = {};
    if (!g1_ik_frame_finish(
            state_candidate,
            result_candidate,
            transaction,
            scratch_positions,
            scratch_rotations,
            bone_parents,
            field,
            dt,
            error,
            error_capacity)) {
        return false;
    }
    if (result_candidate.applied) {
        std::memcpy(
            output_positions.data,
            scratch_positions.data,
            static_cast<std::size_t>(G1_BoneCount) * sizeof(vec3));
        std::memcpy(
            output_rotations.data,
            scratch_rotations.data,
            static_cast<std::size_t>(G1_BoneCount) * sizeof(quat));
        state = state_candidate;
    }
    result = result_candidate;
    return true;
}
