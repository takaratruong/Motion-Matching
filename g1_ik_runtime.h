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
    0x3d23d70au, 0x3d2c0831u, 0x3d343958u, 0x3d3c6a7fu,
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
            !g1_ik_runtime_history_is_valid(state.feet[foot].swing)) {
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

#if defined(__GNUC__) || defined(__clang__)
#define G1_IK_RUNTIME_NOINLINE __attribute__((noinline))
#elif defined(_MSC_VER)
#define G1_IK_RUNTIME_NOINLINE __declspec(noinline)
#else
#define G1_IK_RUNTIME_NOINLINE
#endif

static inline G1_IK_RUNTIME_NOINLINE bool
g1_ik_runtime_compute_foot_centers(
    vec3 output[4],
    const slice1d<vec3> global_positions,
    const slice1d<quat> global_rotations,
    const G1LegConfig& config)
{
    for (int probe = 0; probe < 4; ++probe) {
        const vec3 offset = quat_mul_vec3(
            global_rotations(config.ankle),
            config.foot_sphere_centers_local[probe]);
        const vec3 center = global_positions(config.ankle) + offset;
        if (!g1_ik_vec3_is_runtime_value(offset) ||
            !g1_ik_vec3_is_runtime_value(center)) {
            return false;
        }
        // Preserve the exact staged-FK call-boundary words.  The strict
        // clearance kernel canonicalizes its own public inputs; diagnostics
        // must not rewrite signed zero or any other valid binary32 value.
        output[probe] = center;
    }
    return true;
}

#undef G1_IK_RUNTIME_NOINLINE

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
        if (!g1_ik_runtime_compute_foot_centers(
                centers, global_positions, global_rotations,
                configs[foot]) ||
            !g1_foot_lock_reset(
                candidate.feet[foot].lock,
                global_positions(configs[foot].contact),
                error, error_capacity) ||
            !g1_swing_history_reset(
                candidate.feet[foot].swing,
                centers, error, error_capacity)) {
            return false;
        }
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

static inline bool g1_ik_runtime_materialize_landing_target(
    G1FootTarget& target,
    const G1FootprintFootObservation& footprint,
    const G1LegConfig& config,
    char* error,
    int error_capacity)
{
    float sole_y = 0.0f;
    if (g1_apply_swing_lift_y(
            sole_y,
            footprint.predicted_landing_surface.height,
            config.swing_clearance_m,
            error,
            error_capacity) != G1ClearanceOk) {
        return false;
    }
    target.surface.point = vec3(
        footprint.predicted_landing_sole_center.x,
        footprint.predicted_landing_surface.height,
        footprint.predicted_landing_sole_center.z);
    target.surface.normal =
        footprint.predicted_landing_surface.normal;
    target.sole_center = vec3(
        footprint.predicted_landing_sole_center.x,
        sole_y,
        footprint.predicted_landing_sole_center.z);
    if (!g1_foot_target_is_valid(target)) {
        return g1_ik_runtime_fail(
            error, error_capacity,
            "G1 IK landing materialization produced an invalid target");
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
        const bool lock_was_established =
            candidate.candidate_state.feet[foot_index].lock.locked;
        if (!g1_foot_lock_update(
                candidate.candidate_state.feet[foot_index].lock,
                target,
                field,
                configs[foot_index],
                global_positions(configs[foot_index].contact),
                contacts.data[foot_index],
                dt,
                error,
                error_capacity)) {
            return false;
        }
        // The checked lock owns a continuity-preserving output on its rising
        // edge.  Once established, recorded contact is the exact immutable
        // world lock, independent of later root-level animation changes.
        if (contacts.data[foot_index] && lock_was_established) {
            target.sole_center = candidate.candidate_state
                .feet[foot_index].lock.lock_point;
            if (!g1_foot_target_is_valid(target)) {
                return g1_ik_runtime_fail(
                    error, error_capacity,
                    "G1 IK established contact lock target is invalid");
            }
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
            G1FootTarget target = candidate.candidate_result
                .feet[foot_index].target;
            if (!g1_ik_runtime_materialize_landing_target(
                    target, foot, configs[foot_index],
                    error, error_capacity)) {
                return false;
            }
            candidate.candidate_result.feet[foot_index].target = target;
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
    vec3 position_endpoint,
    vec3 final_endpoint)
{
    return position.applied &&
           position.reachable &&
           !position.correction_limited &&
           !position.safe_stop_requested &&
           g1_ik_contact_residual_is_converged(
               position.contact_residual_m) &&
           orientation.applied &&
           !orientation.correction_limited &&
           !orientation.safe_stop_requested &&
           g1_ik_vec3_bits_equal(
               position_endpoint, final_endpoint);
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
    if (!g1_apply_named_contact_position_ik(
            staged_rotations,
            staged_positions,
            baseline_rotations,
            bone_parents,
            config,
            staged_target.sole_center,
            candidate.position,
            error,
            error_capacity)) {
        return false;
    }

    vec3 position_global_storage[G1_BoneCount];
    quat position_rotation_storage[G1_BoneCount];
    const slice1d<vec3> position_globals(
        G1_BoneCount, position_global_storage);
    const slice1d<quat> position_global_rotations(
        G1_BoneCount, position_rotation_storage);
    if (!g1_ik_checked_forward_kinematics(
            position_globals,
            position_global_rotations,
            staged_positions,
            staged_rotations,
            bone_parents,
            error,
            error_capacity) ||
        !g1_apply_named_foot_orientation(
            staged_rotations,
            staged_positions,
            baseline_rotations,
            bone_parents,
            config,
            staged_target.surface.normal,
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
            position_globals(config.contact),
            final_globals(config.contact));

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
    if (!g1_apply_named_contact_position_ik(
            staged_rotations,
            staged_positions,
            baseline_rotations,
            bone_parents,
            config,
            target.sole_center,
            candidate.position,
            error,
            error_capacity) ||
        !g1_apply_named_foot_orientation(
            staged_rotations,
            staged_positions,
            baseline_rotations,
            bone_parents,
            config,
            target.surface.normal,
            candidate.orientation,
            error,
            error_capacity)) {
        return false;
    }
    vec3 globals_storage[G1_BoneCount];
    quat rotations_storage[G1_BoneCount];
    const slice1d<vec3> globals(
        G1_BoneCount, globals_storage);
    const slice1d<quat> global_rotations(
        G1_BoneCount, rotations_storage);
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
            config)) {
        return false;
    }
    candidate.passes =
        candidate.position.applied &&
        candidate.position.reachable &&
        !candidate.position.correction_limited &&
        !candidate.position.safe_stop_requested &&
        g1_ik_contact_residual_is_converged(
            candidate.position.contact_residual_m) &&
        candidate.orientation.applied &&
        !candidate.orientation.correction_limited &&
        !candidate.orientation.safe_stop_requested;
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
            transaction.candidate_state)) {
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
    if (transaction.candidate_result.applied ||
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
        if (!candidate.candidate_result.feet[foot_index]
                 .recorded_contact) {
            if (selection.selected_index == G1SwingNoCandidate ||
                selection.selected.candidate_index !=
                    selection.selected_index) {
                return g1_ik_runtime_fail(
                    error, error_capacity,
                    "G1 IK final swing selection is incomplete");
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
