#pragma once

#include "g1_candidate_recovery.h"
#include "g1_controller_state.h"
#include "motion_match_log.h"
#include "route_runtime.h"

#include <cfloat>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

enum G1FrameRejectionStage
{
    G1FrameRejectNone = 0,
    G1FrameRejectFootprint,
    G1FrameRejectLandingPatch,
    G1FrameRejectIkCandidate,
    G1FrameRejectPoseCertificate,
};

static inline const char* g1_frame_rejection_stage_name(
    G1FrameRejectionStage stage)
{
    switch (stage) {
    case G1FrameRejectNone: return "none";
    case G1FrameRejectFootprint: return "footprint";
    case G1FrameRejectLandingPatch: return "landing-patch";
    case G1FrameRejectIkCandidate: return "ik-candidate";
    case G1FrameRejectPoseCertificate: return "pose-certificate";
    default: return "unknown";
    }
}

struct G1FrameRejectionDiagnostic
{
    bool rejected = false;
    G1FrameRejectionStage stage = G1FrameRejectNone;
    G1IkStopReason stop_reason = G1IkStopNone;
    bool attempted_footprint_available = false;
    G1FootprintStatus footprint_status = G1FootprintInvalidInput;
    G1FootprintObservation attempted_footprint;
    bool attempted_ik_available = false;
    G1IkFrameResult ik_frame;
    bool attempted_pose_available = false;
    G1ClearanceStatus pose_status = G1ClearanceInvalidInput;
    G1PoseClearance pose_clearance;
};

struct G1FramePublication
{
    G1CommandIntent requested_intent;
    G1FrameRejectionDiagnostic rejection;
    bool ik_safe_stop_latched = false;
    int presentation_frame = 0;
};

struct G1FrameAcceptedDiagnostic
{
    bool ready = false;
    int presentation_frame = 0;
    int scene_frame = 0;
    deterministic_route_sample route;
    traversability_diagnostics traversal;
    float query[31] = {};
    int query_database_frame = -1;
    int query_range = -1;
    int selected_database_frame = -1;
    terrain_centerline_snapshot terrain_query = {};
    motion_match_pose_diagnostic raw_selected;
    motion_match_pose_diagnostic inertialized;
    motion_match_pose_diagnostic support_retargeted;
    motion_match_pose_diagnostic rendered;
    bool matching_enabled = true;
    bool adjustment_enabled = true;
    bool clamping_enabled = true;
    bool ik_enabled = false;
    float effective_terrain_weight = 0.0f;
};

struct G1FrameCandidateWorkspace
{
    g1_controller_state common_state;
    g1_controller_state raw_state;
    g1_controller_state ik_state;
};

struct G1FrameRuntime
{
    g1_controller_state accepted_state;
    g1_controller_state working_state;
    G1FrameCandidateWorkspace candidates;
    G1FramePublication publication;
    G1FrameAcceptedDiagnostic accepted_diagnostic;
};

enum G1FrameTransactionStage
{
    G1FrameStageInputRouteCommand = 0,
    G1FrameStageMatcherSearch,
    G1FrameStageCandidateApply,
    G1FrameStageInertialization,
    G1FrameStageSimulationUpdate,
    G1FrameStageSupportObservation,
    G1FrameStageSupportRetarget,
    G1FrameStageContactUpdate,
    G1FrameStageFootprintObservation,
    G1FrameStageRawBegin,
    G1FrameStageRawFirstFoot,
    G1FrameStageRawSecondFoot,
    G1FrameStageRawFinalFk,
    G1FrameStageRawPoseCertificate,
    G1FrameStageIkBegin,
    G1FrameStageIkFirstFoot,
    G1FrameStageIkSecondFoot,
    G1FrameStageIkFinalFk,
    G1FrameStageIkPoseCertificate,
    G1FrameStageAcceptedFinalize,
    G1FrameStageCount,
};

enum G1FrameStageOutcome
{
    G1FrameStageContinue = 0,
    G1FrameStageFiniteReject,
    G1FrameStageGlobalError,
};

enum G1FrameTransactionStatus
{
    G1FrameTransactionAccepted = 0,
    G1FrameTransactionFiniteRejected,
    G1FrameTransactionGlobalError,
};

enum G1FrameCertificateBranch : uint32_t
{
    G1FrameCertificateNone = 0U,
    G1FrameCertificateRaw,
    G1FrameCertificateIk,
};

struct G1FrameBranchCertificateScratch
{
    G1IkFrameTransaction ik_transaction;
    G1ClearanceStatus pose_status = G1ClearanceInvalidInput;
    G1PoseClearance pose_clearance;
};

struct G1FrameTransactionScratch
{
    bool prior_safe_stop_latched = false;
    bool requested_intent_ready = false;
    G1CommandIntent requested_intent;
    vec3 commanded_velocity;
    vec3 traversal_input;
    G1IkSafeStopHandoff safe_stop_handoff;
    bool force_search = false;
    deterministic_route_sample route_sample;
    traversability_diagnostics traversal;
    float query[31] = {};
    int query_database_frame = -1;
    int query_range = -1;
    int selected_database_frame = -1;
    terrain_centerline_snapshot terrain_query = {};
    motion_match_pose_diagnostic raw_selected_diagnostic;
    motion_match_pose_diagnostic inertialized_diagnostic;
    motion_match_pose_diagnostic support_retargeted_diagnostic;
    motion_match_pose_diagnostic rendered_diagnostic;
    G1FootContactSchedule contact_schedule;
    G1FootprintStatus footprint_status = G1FootprintInvalidInput;
    G1FootprintObservation footprint;
    G1CandidateRecord slot_zero_record;
    G1CandidateRecord active_candidate;
    bool matching_scheduled = false;
    bool legacy_search_performed = false;
    float transition_cost = 0.0f;
    bool recovery_request_ready = false;
    G1RecoveryRequest recovery_request;
    G1FrameBranchCertificateScratch raw_certificate;
    G1FrameBranchCertificateScratch ik_certificate;
    G1FrameCertificateBranch rejection_branch = G1FrameCertificateNone;
    G1FrameRejectionDiagnostic rejection;
    G1FrameAcceptedDiagnostic accepted_diagnostic_candidate;
    bool accepted_diagnostic_ready = false;
};

enum g1_test_mode
{
    G1_TestLive,
    G1_TestSequential,
    G1_TestFlat,
    G1_TestTerrain,
    G1_TestRoute,
    G1_TestSceneCycle,
};

struct G1FrameInputSnapshot
{
    vec3 move_stick;
    vec3 look_stick;
    float gait_target = 0.0f;
    float camera_zoom_axis = 0.0f;
    float scripted_azimuth_delta = 0.0f;
    bool desired_strafe = false;
    int presentation_frame = 0;
};

struct G1FrameTuning
{
    g1_test_mode mode = G1_TestLive;
    int frame_limit = 0;
    int scene_dwell_frames = 25;
    float dt = 1.0f / 25.0f;
    float trajectory_sample_time = 1.0f / 3.0f;
    float route_speed = 0.50f;
    float future_speed_scale = 1.0f;
    float walkability_radius = 0.20f;
    float effective_terrain_weight = 0.0f;
    float initial_search_time = 0.10f;
    float inertialize_blending_halflife = 0.10f;
    float desired_velocity_change_threshold = 50.0f;
    float desired_rotation_change_threshold = 50.0f;
    float simulation_velocity_halflife = 0.27f;
    float simulation_rotation_halflife = 0.27f;
    float simulation_run_forward_speed = 0.90f;
    float simulation_run_side_speed = 0.60f;
    float simulation_run_back_speed = 0.60f;
    float simulation_walk_forward_speed = 0.50f;
    float simulation_walk_side_speed = 0.40f;
    float simulation_walk_back_speed = 0.40f;
    bool synchronization_enabled = false;
    float synchronization_data_factor = 1.0f;
    bool adjustment_enabled = true;
    bool adjustment_by_velocity_enabled = true;
    float adjustment_position_halflife = 0.10f;
    float adjustment_rotation_halflife = 0.20f;
    float adjustment_position_max_ratio = 0.50f;
    float adjustment_rotation_max_ratio = 0.50f;
    bool clamping_enabled = true;
    float clamping_max_distance = 0.15f;
    float clamping_max_angle = 0.5f * PIf;
    float contact_unlock_radius = 0.20f;
    float contact_foot_height = 0.02f;
    float contact_blending_halflife = 0.10f;
    bool ik_enabled = false;
};

struct G1FrameExternalInputs
{
    const database* db = nullptr;
    const terrain_support_set* support = nullptr;
    const scene_pack* scene = nullptr;
    const scene_route* route = nullptr;
    G1TestHeadingOverride heading_override;
    G1FrameInputSnapshot input;
    G1FrameTuning tuning;
};

struct G1FrameResetConfig
{
    bool route_mode = false;
    const char* route_id = nullptr;
    bool ik_enabled = false;
    float dt = 1.0f / 25.0f;
    float trajectory_sample_time = 1.0f / 3.0f;
    float initial_search_time = 0.10f;
};

using G1FrameStageRunner = G1FrameStageOutcome (*)(
    G1FrameTransactionStage stage,
    g1_controller_state& working_state,
    G1FrameTransactionScratch& scratch,
    const G1FrameExternalInputs& external,
    char* error,
    int error_capacity);

#if defined(G1_FRAME_TRANSACTION_ENABLE_TEST_SEAM)
enum G1FrameInjectedOutcome
{
    G1FrameInjectContinue = 0,
    G1FrameInjectFiniteReject,
    G1FrameInjectGlobalError,
};

struct G1FrameTransactionTestControl
{
    G1FrameTransactionStage injected_stage = G1FrameStageCount;
    G1FrameInjectedOutcome injected_outcome = G1FrameInjectContinue;
};

enum G1CandidateScoreOwner : uint32_t
{
    G1CandidateScoreUnassigned = 0U,
    G1CandidateScoreLegacy,
    G1CandidateScoreStrictRecovery,
    G1CandidateScoreIncumbent,
};

enum G1CandidateDisposition : uint32_t
{
    G1CandidateDispositionNotRun = 0U,
    G1CandidateDispositionAccepted,
    G1CandidateDispositionFiniteRejected,
    G1CandidateDispositionGlobalError,
};

struct G1CandidateAttemptTraceRecord
{
    G1CandidateRecord candidate;
    G1CandidateScoreOwner score_owner = G1CandidateScoreUnassigned;
    G1CandidateDisposition common = G1CandidateDispositionNotRun;
    G1CandidateDisposition raw = G1CandidateDispositionNotRun;
    G1CandidateDisposition ik = G1CandidateDispositionNotRun;
    G1FrameRejectionStage rejection_stage = G1FrameRejectNone;
    G1IkStopReason stop_reason = G1IkStopNone;
};

struct G1CandidateCertificationTrace
{
    G1CandidateAttemptTraceRecord attempts[G1CandidateAttemptCapacity] = {};
    uint32_t attempt_count = 0U;
    uint32_t legacy_traversals = 0U;
    uint32_t recovery_provider_calls = 0U;
    uint32_t common_evaluations = 0U;
    uint32_t raw_evaluations = 0U;
    uint32_t ik_evaluations = 0U;
    bool recovery_request_available = false;
    G1RecoveryRequest recovery_request;
    G1RecoveryCandidateSet recovery_set;
};

static inline void g1_frame_candidate_record_reset(
    G1CandidateRecord& record)
{
    record.kind = G1CandidateIncumbent;
    record.selected_frame = -1;
    record.executed_frame = -1;
    record.source_range = -1;
    record.selected_cost = FLT_MAX;
    record.recovery_rank = UINT32_MAX;
    record.transitioned = false;
}

static inline void g1_frame_attempt_trace_reset(
    G1CandidateAttemptTraceRecord& record)
{
    g1_frame_candidate_record_reset(record.candidate);
    record.score_owner = G1CandidateScoreUnassigned;
    record.common = G1CandidateDispositionNotRun;
    record.raw = G1CandidateDispositionNotRun;
    record.ik = G1CandidateDispositionNotRun;
    record.rejection_stage = G1FrameRejectNone;
    record.stop_reason = G1IkStopNone;
}

static inline void g1_frame_recovery_request_reset(
    G1RecoveryRequest& request)
{
    request.db = nullptr;
    for (uint32_t feature = 0U;
         feature < G1RecoveryFeatureCount;
         ++feature) {
        request.raw_query[feature] = 0.0f;
    }
    request.incumbent_frame = -1;
    request.legacy_selected_frame = -1;
    request.transition_cost = 0.0f;
    request.public_incumbent_cost = FLT_MAX;
    request.ignore_range_end = 20;
    request.ignore_surrounding = 20;
}

static inline void g1_frame_recovery_set_reset(
    G1RecoveryCandidateSet& set)
{
    for (uint32_t index = 0U;
         index < G1CandidateAttemptCapacity;
         ++index) {
        g1_frame_candidate_record_reset(set.records[index]);
    }
    set.count = 0U;
    set.work.accelerated_traversals = 0U;
    set.work.large_bounds_tested = 0U;
    set.work.small_bounds_tested = 0U;
    set.work.rows_tested = 0U;
    set.work.full_scores_materialized = 0U;
}

static inline void g1_frame_certification_trace_reset(
    G1CandidateCertificationTrace& trace)
{
    for (uint32_t index = 0U;
         index < G1CandidateAttemptCapacity;
         ++index) {
        g1_frame_attempt_trace_reset(trace.attempts[index]);
    }
    trace.attempt_count = 0U;
    trace.legacy_traversals = 0U;
    trace.recovery_provider_calls = 0U;
    trace.common_evaluations = 0U;
    trace.raw_evaluations = 0U;
    trace.ik_evaluations = 0U;
    trace.recovery_request_available = false;
    g1_frame_recovery_request_reset(trace.recovery_request);
    g1_frame_recovery_set_reset(trace.recovery_set);
}

using G1FrameTransactionTestHook = G1FrameInjectedOutcome (*)(
    G1FrameTransactionStage stage,
    const g1_controller_state& working_state,
    G1FrameTransactionScratch& scratch,
    const G1FrameExternalInputs& external,
    const G1FrameTransactionTestControl& control,
    char* error,
    int error_capacity);

struct G1FrameTransactionTestSeam
{
    G1FrameTransactionTestHook hook = nullptr;
    G1FrameTransactionTestControl control;
    G1CandidateCertificationTrace* certification_trace = nullptr;
};
#endif

static inline bool g1_frame_float_bits_equal(float first, float second)
{
    return terrain_float_bits(first) == terrain_float_bits(second);
}

static inline uint64_t g1_frame_double_bits(double value)
{
    uint64_t bits = 0U;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static inline bool g1_frame_double_bits_equal(double first, double second)
{
    return g1_frame_double_bits(first) == g1_frame_double_bits(second);
}

static inline bool g1_frame_vec3_bits_equal(vec3 first, vec3 second)
{
    return g1_frame_float_bits_equal(first.x, second.x) &&
           g1_frame_float_bits_equal(first.y, second.y) &&
           g1_frame_float_bits_equal(first.z, second.z);
}

static inline bool g1_frame_quat_bits_equal(quat first, quat second)
{
    return g1_frame_float_bits_equal(first.w, second.w) &&
           g1_frame_float_bits_equal(first.x, second.x) &&
           g1_frame_float_bits_equal(first.y, second.y) &&
           g1_frame_float_bits_equal(first.z, second.z);
}

template<class T>
static inline bool g1_frame_array_values_equal(
    const array1d<T>& first,
    const array1d<T>& second)
{
    if (first.size != second.size || first.size < 0) return false;
    if (first.size == 0) {
        return first.data == nullptr && second.data == nullptr;
    }
    if (first.data == nullptr || second.data == nullptr ||
        static_cast<std::size_t>(first.size) >
            std::numeric_limits<std::size_t>::max() / sizeof(T)) {
        return false;
    }
    return std::memcmp(
               first.data,
               second.data,
               static_cast<std::size_t>(first.size) * sizeof(T)) == 0;
}

static inline bool g1_frame_clearance_work_equal(
    const G1ClearanceWork& first,
    const G1ClearanceWork& second)
{
    return first.point_queries == second.point_queries &&
           first.cells_visited == second.cells_visited &&
           first.primitive_triangle_pairs ==
               second.primitive_triangle_pairs &&
           first.face_patches == second.face_patches &&
           first.candidate_tests == second.candidate_tests &&
           first.subdivision_nodes == second.subdivision_nodes;
}

static inline bool g1_frame_surface_equal(
    const G1SurfaceSample& first,
    const G1SurfaceSample& second)
{
    return g1_frame_float_bits_equal(first.height, second.height) &&
           g1_frame_vec3_bits_equal(first.normal, second.normal);
}

static inline bool g1_frame_target_equal(
    const G1FootTarget& first,
    const G1FootTarget& second)
{
    return first.locked == second.locked &&
           first.position_active == second.position_active &&
           first.releasing == second.releasing &&
           first.drift_limit_exceeded == second.drift_limit_exceeded &&
           g1_frame_vec3_bits_equal(
               first.surface.point, second.surface.point) &&
           g1_frame_vec3_bits_equal(
               first.surface.normal, second.surface.normal) &&
           g1_frame_vec3_bits_equal(
               first.desired_sole_normal,
               second.desired_sole_normal) &&
           g1_frame_vec3_bits_equal(first.sole_center, second.sole_center) &&
           g1_frame_float_bits_equal(
               first.horizontal_drift_m, second.horizontal_drift_m);
}

static inline bool g1_frame_swing_candidate_equal(
    const G1SwingCandidateDiagnostic& first,
    const G1SwingCandidateDiagnostic& second)
{
    if (first.candidate_index != second.candidate_index ||
        first.lift_bits != second.lift_bits ||
        first.materialized_command_y_bits !=
            second.materialized_command_y_bits ||
        first.clearance_status != second.clearance_status ||
        first.controller_constraints_passed !=
            second.controller_constraints_passed ||
        first.clearance_certified != second.clearance_certified ||
        !g1_frame_double_bits_equal(
            first.lower_margin_m, second.lower_margin_m) ||
        !g1_frame_double_bits_equal(
            first.witness_upper_margin_m,
            second.witness_upper_margin_m) ||
        !g1_frame_clearance_work_equal(
            first.clearance_work, second.clearance_work)) {
        return false;
    }
    for (int probe = 0; probe < 4; ++probe) {
        for (int axis = 0; axis < 3; ++axis) {
            if (first.actual_sphere_center_bits[probe][axis] !=
                second.actual_sphere_center_bits[probe][axis]) {
                return false;
            }
        }
    }
    return true;
}

static inline bool g1_frame_leg_result_equal(
    const G1LegSolveResult& first,
    const G1LegSolveResult& second)
{
    return first.applied == second.applied &&
           first.reachable == second.reachable &&
           first.correction_limited == second.correction_limited &&
           first.safe_stop_requested == second.safe_stop_requested &&
           first.iterations == second.iterations &&
           first.iteration_provenance ==
               second.iteration_provenance &&
           g1_frame_vec3_bits_equal(
               first.requested_ankle_target,
               second.requested_ankle_target) &&
           g1_frame_vec3_bits_equal(
               first.clamped_ankle_target,
               second.clamped_ankle_target) &&
           g1_frame_vec3_bits_equal(
               first.hinge_axis_world, second.hinge_axis_world) &&
           g1_frame_vec3_bits_equal(
               first.bend_direction, second.bend_direction) &&
           first.bend_used_current_projection ==
               second.bend_used_current_projection &&
           first.bend_used_hinge_fallback ==
               second.bend_used_hinge_fallback &&
           first.bend_used_safe_perpendicular ==
               second.bend_used_safe_perpendicular &&
           first.bend_sign_flipped == second.bend_sign_flipped &&
           g1_frame_float_bits_equal(
               first.raw_distance_m, second.raw_distance_m) &&
           g1_frame_float_bits_equal(
               first.clamped_distance_m, second.clamped_distance_m) &&
           g1_frame_float_bits_equal(
               first.max_correction_radians,
               second.max_correction_radians) &&
           g1_frame_float_bits_equal(
               first.contact_residual_m, second.contact_residual_m);
}

static inline bool g1_frame_orientation_equal(
    const G1FootOrientationResult& first,
    const G1FootOrientationResult& second)
{
    return first.applied == second.applied &&
           first.correction_limited == second.correction_limited &&
           first.safe_stop_requested == second.safe_stop_requested &&
           g1_frame_quat_bits_equal(
               first.target_global_rotation,
               second.target_global_rotation) &&
           g1_frame_float_bits_equal(
               first.requested_correction_radians,
               second.requested_correction_radians) &&
           g1_frame_float_bits_equal(
               first.correction_radians, second.correction_radians);
}

static inline bool g1_frame_ik_result_equal(
    const G1IkFrameResult& first,
    const G1IkFrameResult& second)
{
    if (first.applied != second.applied ||
        first.safe_stop_requested != second.safe_stop_requested ||
        first.stop_reason != second.stop_reason ||
        !g1_frame_float_bits_equal(
            first.max_correction_radians,
            second.max_correction_radians) ||
        first.root_reach.active != second.root_reach.active ||
        first.root_reach.common_interval_found !=
            second.root_reach.common_interval_found ||
        first.root_reach.applied != second.root_reach.applied ||
        !g1_frame_float_bits_equal(
            first.root_reach.root_y_delta_m,
            second.root_reach.root_y_delta_m)) {
        return false;
    }
    for (int foot = 0; foot < 2; ++foot) {
        const G1FootFrameResult& left = first.feet[foot];
        const G1FootFrameResult& right = second.feet[foot];
        if (left.recorded_contact != right.recorded_contact ||
            !g1_frame_target_equal(left.target, right.target) ||
            left.swing_selection.candidates_evaluated !=
                right.swing_selection.candidates_evaluated ||
            left.swing_selection.selected_index !=
                right.swing_selection.selected_index ||
            !g1_frame_swing_candidate_equal(
                left.swing_selection.selected,
                right.swing_selection.selected) ||
            !g1_frame_clearance_work_equal(
                left.swing_selection.total_clearance_work,
                right.swing_selection.total_clearance_work) ||
            !g1_frame_double_bits_equal(
                left.defensive_swing.lower_margin_m,
                right.defensive_swing.lower_margin_m) ||
            !g1_frame_double_bits_equal(
                left.defensive_swing.witness_upper_m,
                right.defensive_swing.witness_upper_m) ||
            left.defensive_swing.sweep_evaluated !=
                right.defensive_swing.sweep_evaluated ||
            !g1_frame_clearance_work_equal(
                left.defensive_swing.work,
                right.defensive_swing.work) ||
            !g1_frame_leg_result_equal(left.position, right.position) ||
            !g1_frame_orientation_equal(
                left.orientation, right.orientation)) {
            return false;
        }
    }
    return true;
}

static inline bool g1_frame_ik_state_equal(
    const G1IkState& first,
    const G1IkState& second)
{
    if (first.initialized != second.initialized) return false;
    for (int foot = 0; foot < 2; ++foot) {
        const G1FootLockState& left = first.feet[foot].lock;
        const G1FootLockState& right = second.feet[foot].lock;
        if (left.initialized != right.initialized ||
            left.contact != right.contact ||
            left.locked != right.locked ||
            left.position_active != right.position_active ||
            left.releasing != right.releasing ||
            left.release_frames != right.release_frames ||
            !g1_frame_vec3_bits_equal(
                left.previous_input, right.previous_input) ||
            !g1_frame_vec3_bits_equal(left.lock_point, right.lock_point) ||
            !g1_frame_vec3_bits_equal(
                left.output_position, right.output_position) ||
            !g1_frame_vec3_bits_equal(
                left.output_velocity, right.output_velocity) ||
            !g1_frame_vec3_bits_equal(
                left.offset_position, right.offset_position) ||
            !g1_frame_vec3_bits_equal(
                left.offset_velocity, right.offset_velocity) ||
            first.feet[foot].swing.initialized !=
                second.feet[foot].swing.initialized ||
            !g1_frame_vec3_bits_equal(
                first.feet[foot].baseline_sole_normal,
                second.feet[foot].baseline_sole_normal)) {
            return false;
        }
        for (int probe = 0; probe < 4; ++probe) {
            if (!g1_frame_vec3_bits_equal(
                    first.feet[foot].swing
                        .previous_sphere_centers[probe],
                    second.feet[foot].swing
                        .previous_sphere_centers[probe])) {
                return false;
            }
        }
    }
    return true;
}

static inline bool g1_frame_footprint_equal(
    const G1FootprintObservation& first,
    const G1FootprintObservation& second)
{
    if (!g1_frame_surface_equal(
            first.root_surface, second.root_surface) ||
        first.blocked != second.blocked ||
        first.blocked_reason != second.blocked_reason ||
        first.work.sweeps != second.work.sweeps ||
        first.work.surface_queries != second.work.surface_queries ||
        first.work.node_visits != second.work.node_visits) {
        return false;
    }
    for (int foot = 0; foot < 2; ++foot) {
        const G1FootprintFootObservation& left = first.feet[foot];
        const G1FootprintFootObservation& right = second.feet[foot];
        if (left.current_contact != right.current_contact ||
            left.landing_expected != right.landing_expected ||
            left.landing_patch_ready != right.landing_patch_ready ||
            left.landing_sample != right.landing_sample ||
            !g1_frame_vec3_bits_equal(
                left.predicted_landing_sole_center,
                right.predicted_landing_sole_center) ||
            left.predicted_landing_surface_status !=
                right.predicted_landing_surface_status ||
            !g1_frame_surface_equal(
                left.predicted_landing_surface,
                right.predicted_landing_surface) ||
            left.predicted_landing_walkability_class !=
                right.predicted_landing_walkability_class ||
            !g1_frame_double_bits_equal(
                left.landing_patch_maximum_residual_m,
                right.landing_patch_maximum_residual_m) ||
            !g1_frame_float_bits_equal(
                left.corridor_minimum_height,
                right.corridor_minimum_height) ||
            !g1_frame_float_bits_equal(
                left.corridor_maximum_height,
                right.corridor_maximum_height) ||
            !g1_frame_double_bits_equal(
                left.maximum_root_split_m,
                right.maximum_root_split_m) ||
            left.encountered_walkability_class !=
                right.encountered_walkability_class ||
            left.multilevel != right.multilevel) {
            return false;
        }
        for (int probe = 0; probe < 4; ++probe) {
            const G1FootprintProbe& left_probe = left.probes[probe];
            const G1FootprintProbe& right_probe = right.probes[probe];
            if (!g1_frame_vec3_bits_equal(
                    left_probe.current_sphere_center,
                    right_probe.current_sphere_center) ||
                !g1_frame_vec3_bits_equal(
                    left_probe.current_sole_point,
                    right_probe.current_sole_point) ||
                !g1_frame_surface_equal(
                    left_probe.current_surface,
                    right_probe.current_surface) ||
                !g1_frame_surface_equal(
                    left_probe.selected_landing_surface,
                    right_probe.selected_landing_surface) ||
                !g1_frame_float_bits_equal(
                    left_probe.corridor_minimum_height,
                    right_probe.corridor_minimum_height) ||
                !g1_frame_float_bits_equal(
                    left_probe.corridor_maximum_height,
                    right_probe.corridor_maximum_height) ||
                left_probe.encountered_walkability_class !=
                    right_probe.encountered_walkability_class) {
                return false;
            }
            for (int sample = 0;
                 sample < G1CommandTrajectorySampleCount;
                 ++sample) {
                if (!g1_frame_vec3_bits_equal(
                        left_probe.predicted_sphere_centers[sample],
                        right_probe.predicted_sphere_centers[sample]) ||
                    !g1_frame_vec3_bits_equal(
                        left_probe.predicted_sole_points[sample],
                        right_probe.predicted_sole_points[sample]) ||
                    left_probe.predicted_surface_status[sample] !=
                        right_probe.predicted_surface_status[sample] ||
                    !g1_frame_surface_equal(
                        left_probe.predicted_surfaces[sample],
                        right_probe.predicted_surfaces[sample])) {
                    return false;
                }
            }
        }
    }
    return true;
}

static inline bool g1_frame_command_equal(
    const G1CommandSnapshot& first,
    const G1CommandSnapshot& second)
{
    if (!g1_frame_vec3_bits_equal(
            first.intent.requested_velocity,
            second.intent.requested_velocity) ||
        !g1_frame_quat_bits_equal(
            first.intent.desired_heading,
            second.intent.desired_heading) ||
        !g1_frame_vec3_bits_equal(
            first.applied_velocity, second.applied_velocity)) {
        return false;
    }
    for (int sample = 0;
         sample < G1CommandTrajectorySampleCount;
         ++sample) {
        if (!g1_frame_vec3_bits_equal(
                first.predicted_desired_velocities[sample],
                second.predicted_desired_velocities[sample]) ||
            !g1_frame_vec3_bits_equal(
                first.predicted_root_positions[sample],
                second.predicted_root_positions[sample]) ||
            !g1_frame_quat_bits_equal(
                first.predicted_root_rotations[sample],
                second.predicted_root_rotations[sample]) ||
            !g1_frame_quat_bits_equal(
                first.predicted_desired_headings[sample],
                second.predicted_desired_headings[sample])) {
            return false;
        }
    }
    return true;
}

static inline bool g1_frame_state_range_sets_are_disjoint(
    const g1_controller_state& first_state,
    const g1_controller_state_memory_range* first,
    int first_count,
    const g1_controller_state& second_state,
    const g1_controller_state_memory_range* second,
    int second_count)
{
    if (first == nullptr || second == nullptr ||
        first_count < 0 || second_count < 0 ||
        g1_ik_memory_ranges_overlap(
            &first_state, sizeof(first_state),
            &second_state, sizeof(second_state))) {
        return false;
    }
    const g1_controller_state_memory_range first_object = {
        &first_state, sizeof(first_state)
    };
    const g1_controller_state_memory_range second_object = {
        &second_state, sizeof(second_state)
    };
    for (int first_index = 0; first_index < first_count; ++first_index) {
        if (g1_controller_state_ranges_overlap(
                first[first_index], second_object)) {
            return false;
        }
        for (int second_index = 0;
             second_index < second_count;
             ++second_index) {
            if (g1_controller_state_ranges_overlap(
                    first[first_index], second[second_index])) {
                return false;
            }
        }
    }
    for (int second_index = 0;
         second_index < second_count;
         ++second_index) {
        if (g1_controller_state_ranges_overlap(
                second[second_index], first_object)) {
            return false;
        }
    }
    return true;
}

static inline bool g1_frame_state_pair_storage_is_exact(
    const g1_controller_state& first,
    const g1_controller_state& second,
    g1_controller_state_memory_range* first_ranges,
    int& first_count,
    g1_controller_state_memory_range* second_ranges,
    int& second_count,
    int range_capacity)
{
    return g1_controller_state_storage_ranges(
               first,
               first_ranges,
               first_count,
               range_capacity) &&
           g1_controller_state_storage_ranges(
               second,
               second_ranges,
               second_count,
               range_capacity) &&
           first_count == 49 && second_count == 49 &&
           g1_controller_state_ranges_are_disjoint(
               first_ranges, first_count) &&
           g1_controller_state_ranges_are_disjoint(
               second_ranges, second_count) &&
           g1_frame_state_range_sets_are_disjoint(
               first,
               first_ranges,
               first_count,
               second,
               second_ranges,
               second_count);
}

static inline bool g1_frame_state_pair_reset_storage_is_safe(
    const g1_controller_state& first,
    const g1_controller_state& second,
    g1_controller_state_memory_range* first_ranges,
    int& first_count,
    g1_controller_state_memory_range* second_ranges,
    int& second_count,
    int range_capacity)
{
    return g1_controller_state_reset_output_ranges(
               first,
               first_ranges,
               first_count,
               range_capacity) &&
           g1_controller_state_reset_output_ranges(
               second,
               second_ranges,
               second_count,
               range_capacity) &&
           g1_frame_state_range_sets_are_disjoint(
               first,
               first_ranges,
               first_count,
               second,
               second_ranges,
               second_count);
}

static inline void g1_frame_runtime_state_pointers(
    const G1FrameRuntime& runtime,
    const g1_controller_state* states[5])
{
    states[0] = &runtime.accepted_state;
    states[1] = &runtime.working_state;
    states[2] = &runtime.candidates.common_state;
    states[3] = &runtime.candidates.raw_state;
    states[4] = &runtime.candidates.ik_state;
}

static inline bool g1_frame_runtime_storage_sets_are_safe(
    const G1FrameRuntime& runtime,
    bool require_exact_storage,
    g1_controller_state_memory_range ranges[5][64],
    int counts[5])
{
    const g1_controller_state* states[5] = {};
    g1_frame_runtime_state_pointers(runtime, states);
    for (int state_index = 0; state_index < 5; ++state_index) {
        const bool collected = require_exact_storage
            ? g1_controller_state_storage_ranges(
                  *states[state_index],
                  ranges[state_index],
                  counts[state_index],
                  64)
            : g1_controller_state_reset_output_ranges(
                  *states[state_index],
                  ranges[state_index],
                  counts[state_index],
                  64);
        if (!collected ||
            (require_exact_storage &&
             (counts[state_index] != 49 ||
              !g1_controller_state_ranges_are_disjoint(
                  ranges[state_index], counts[state_index])))) {
            return false;
        }
    }
    for (int first = 0; first < 5; ++first) {
        for (int second = first + 1; second < 5; ++second) {
            if (states[first] == states[second] ||
                !g1_frame_state_range_sets_are_disjoint(
                    *states[first],
                    ranges[first],
                    counts[first],
                    *states[second],
                    ranges[second],
                    counts[second])) {
                return false;
            }
        }
    }
    return true;
}

static inline bool g1_frame_error_overlaps_ranges(
    char* error,
    int error_capacity,
    const g1_controller_state_memory_range* ranges,
    int count)
{
    if (error_capacity < 0 || ranges == nullptr || count < 0) return true;
    if (error == nullptr || error_capacity == 0) return false;
    const g1_controller_state_memory_range diagnostic = {
        error, static_cast<std::size_t>(error_capacity)
    };
    for (int index = 0; index < count; ++index) {
        if (g1_controller_state_ranges_overlap(
                diagnostic, ranges[index])) {
            return true;
        }
    }
    return false;
}

static inline bool g1_frame_error_overlaps_object(
    char* error,
    int error_capacity,
    const void* object,
    std::size_t bytes)
{
    if (error_capacity < 0) return true;
    if (error == nullptr || error_capacity == 0) return false;
    return g1_ik_memory_ranges_overlap(
        error,
        static_cast<std::size_t>(error_capacity),
        object,
        bytes);
}

static inline bool g1_frame_database_shapes_are_valid(
    const database& db,
    const terrain_support_set& support)
{
    const int frames = db.bone_positions.rows;
    if (frames <= 0 ||
        db.bone_positions.cols != G1_BoneCount ||
        db.bone_positions.data == nullptr ||
        db.bone_velocities.rows != frames ||
        db.bone_velocities.cols != G1_BoneCount ||
        db.bone_velocities.data == nullptr ||
        db.bone_rotations.rows != frames ||
        db.bone_rotations.cols != G1_BoneCount ||
        db.bone_rotations.data == nullptr ||
        db.bone_angular_velocities.rows != frames ||
        db.bone_angular_velocities.cols != G1_BoneCount ||
        db.bone_angular_velocities.data == nullptr ||
        db.contact_states.rows != frames ||
        db.contact_states.cols != 2 ||
        db.contact_states.data == nullptr ||
        db.bone_parents.size != G1_BoneCount ||
        db.bone_parents.data == nullptr ||
        db.range_starts.size <= 0 ||
        db.range_starts.size != db.range_stops.size ||
        db.range_starts.data == nullptr ||
        db.range_stops.data == nullptr ||
        db.features.rows != frames ||
        db.features.cols != 31 ||
        db.features.data == nullptr ||
        db.features_offset.size != 31 ||
        db.features_offset.data == nullptr ||
        db.features_scale.size != 31 ||
        db.features_scale.data == nullptr ||
        db.terrain_features.rows != frames ||
        db.terrain_features.cols != 4 ||
        db.terrain_features.data == nullptr ||
        support.values.rows != frames ||
        support.values.cols != 3 ||
        support.values.data == nullptr) {
        return false;
    }
    const int small_bounds = (frames - 1) / BOUND_SM_SIZE + 1;
    const int large_bounds = (frames - 1) / BOUND_LR_SIZE + 1;
    if (db.bound_sm_min.rows != small_bounds ||
        db.bound_sm_min.cols != 31 ||
        db.bound_sm_min.data == nullptr ||
        db.bound_sm_max.rows != small_bounds ||
        db.bound_sm_max.cols != 31 ||
        db.bound_sm_max.data == nullptr ||
        db.bound_lr_min.rows != large_bounds ||
        db.bound_lr_min.cols != 31 ||
        db.bound_lr_min.data == nullptr ||
        db.bound_lr_max.rows != large_bounds ||
        db.bound_lr_max.cols != 31 ||
        db.bound_lr_max.data == nullptr) {
        return false;
    }
    int prior_stop = 0;
    for (int range = 0; range < db.range_starts.size; ++range) {
        const int start = db.range_starts(range);
        const int stop = db.range_stops(range);
        if (start != prior_stop || start < 0 ||
            start >= stop || stop > frames) {
            return false;
        }
        prior_stop = stop;
    }
    if (prior_stop != frames) return false;
    for (int dimension = 0; dimension < 31; ++dimension) {
        if (!terrain_float_is_normal_or_zero_query(
                db.features_offset(dimension)) ||
            !terrain_float_is_positive_normal(
                db.features_scale(dimension))) {
            return false;
        }
    }
    return true;
}

static inline bool g1_frame_scene_shape_is_valid(const scene_pack& scene)
{
    const vec3 spawn = scene.metadata.spawn_position;
    return scene.terrain.version == 2 &&
           terrain_heightfield_is_queryable(scene.terrain) &&
           walkability_grid_matches_heightfield(
               scene.walkability, scene.terrain) &&
           scene_bounds2_valid(scene.metadata.playable_bounds) &&
           g1_ik_vec3_is_runtime_value(spawn) &&
           terrain_float_is_normal_or_zero_query(
               scene.metadata.spawn_yaw) &&
           scene_inside(
               scene.metadata.playable_bounds,
               spawn.x,
               spawn.z);
}

static inline bool g1_frame_artifacts_are_valid(
    const database& db,
    const terrain_support_set& support,
    const scene_pack& scene,
    char* error,
    int error_capacity)
{
    return g1_frame_database_shapes_are_valid(db, support) &&
           g1_frame_scene_shape_is_valid(scene) &&
           g1_ik_parent_topology_validate(
               db.bone_parents,
               error,
               error_capacity);
}

static inline bool g1_frame_input_snapshot_is_valid(
    const G1FrameInputSnapshot& input,
    const G1TestHeadingOverride& heading_override)
{
    const vec3 sticks[] = {input.move_stick, input.look_stick};
    for (vec3 value : sticks) {
        if (!g1_ik_vec3_is_runtime_value(value) ||
            value.x < -1.0f || value.x > 1.0f ||
            value.y < -1.0f || value.y > 1.0f ||
            value.z < -1.0f || value.z > 1.0f) {
            return false;
        }
    }
    return terrain_float_is_normal_or_zero_query(input.gait_target) &&
           input.gait_target >= 0.0f && input.gait_target <= 1.0f &&
           terrain_float_is_normal_or_zero_query(
               input.camera_zoom_axis) &&
           terrain_float_is_normal_or_zero_query(
               input.scripted_azimuth_delta) &&
           input.presentation_frame >= 0 &&
           ik_quat_is_unit(heading_override.heading);
}

static inline bool g1_frame_nonnegative_runtime_value(float value)
{
    return terrain_float_is_normal_or_zero_query(value) && value >= 0.0f;
}

static inline bool g1_frame_unit_interval_runtime_value(float value)
{
    return g1_frame_nonnegative_runtime_value(value) && value <= 1.0f;
}

static inline bool g1_frame_tuning_is_valid(const G1FrameTuning& tuning)
{
    if (tuning.mode < G1_TestLive || tuning.mode > G1_TestSceneCycle ||
        tuning.frame_limit < 0 || tuning.scene_dwell_frames <= 0 ||
        (tuning.mode != G1_TestLive && tuning.frame_limit == 0) ||
        !g1_dt_is_exact_25_hz(tuning.dt) ||
        !terrain_float_is_positive_normal(
            tuning.trajectory_sample_time) ||
        !terrain_float_is_positive_normal(tuning.route_speed) ||
        !g1_frame_unit_interval_runtime_value(
            tuning.future_speed_scale) ||
        !g1_frame_nonnegative_runtime_value(
            tuning.walkability_radius) ||
        !g1_frame_nonnegative_runtime_value(
            tuning.effective_terrain_weight) ||
        tuning.effective_terrain_weight > 10.0f ||
        !terrain_float_is_finite(tuning.initial_search_time) ||
        tuning.initial_search_time < 0.0f ||
        tuning.initial_search_time > 10.0f ||
        !terrain_float_is_positive_normal(
            tuning.inertialize_blending_halflife) ||
        !g1_frame_nonnegative_runtime_value(
            tuning.desired_velocity_change_threshold) ||
        !g1_frame_nonnegative_runtime_value(
            tuning.desired_rotation_change_threshold) ||
        !terrain_float_is_positive_normal(
            tuning.simulation_velocity_halflife) ||
        !terrain_float_is_positive_normal(
            tuning.simulation_rotation_halflife) ||
        !g1_frame_nonnegative_runtime_value(
            tuning.simulation_run_forward_speed) ||
        !g1_frame_nonnegative_runtime_value(
            tuning.simulation_run_side_speed) ||
        !g1_frame_nonnegative_runtime_value(
            tuning.simulation_run_back_speed) ||
        !g1_frame_nonnegative_runtime_value(
            tuning.simulation_walk_forward_speed) ||
        !g1_frame_nonnegative_runtime_value(
            tuning.simulation_walk_side_speed) ||
        !g1_frame_nonnegative_runtime_value(
            tuning.simulation_walk_back_speed) ||
        !g1_frame_nonnegative_runtime_value(
            tuning.synchronization_data_factor) ||
        !terrain_float_is_positive_normal(
            tuning.adjustment_position_halflife) ||
        !terrain_float_is_positive_normal(
            tuning.adjustment_rotation_halflife) ||
        !g1_frame_unit_interval_runtime_value(
            tuning.adjustment_position_max_ratio) ||
        !g1_frame_unit_interval_runtime_value(
            tuning.adjustment_rotation_max_ratio) ||
        !g1_frame_nonnegative_runtime_value(
            tuning.clamping_max_distance) ||
        !g1_frame_nonnegative_runtime_value(
            tuning.clamping_max_angle) ||
        !g1_frame_nonnegative_runtime_value(
            tuning.contact_unlock_radius) ||
        !g1_frame_nonnegative_runtime_value(
            tuning.contact_foot_height) ||
        !terrain_float_is_positive_normal(
            tuning.contact_blending_halflife)) {
        return false;
    }
    if (tuning.mode == G1_TestSceneCycle) {
        if (tuning.scene_dwell_frames > INT_MAX / 14) return false;
        const int minimum_frames = 14 * tuning.scene_dwell_frames;
        if (tuning.frame_limit < minimum_frames) return false;
    }
    return true;
}

static inline int g1_frame_route_member_index(
    const scene_pack& scene,
    const scene_route* route)
{
    if (route == nullptr) return -1;
    for (std::size_t index = 0;
         index < scene.metadata.routes.size();
         ++index) {
        if (route == &scene.metadata.routes[index]) {
            if (index > static_cast<std::size_t>(INT_MAX)) return -1;
            return static_cast<int>(index);
        }
    }
    return -1;
}

static inline int g1_frame_route_id_index(
    const scene_pack& scene,
    const char* route_id)
{
    if (route_id == nullptr || route_id[0] == '\0') return -1;
    for (std::size_t index = 0;
         index < scene.metadata.routes.size();
         ++index) {
        if (scene.metadata.routes[index].id == route_id) {
            if (index > static_cast<std::size_t>(INT_MAX)) return -1;
            return static_cast<int>(index);
        }
    }
    return -1;
}

static inline bool g1_frame_route_relation_is_valid(
    const G1FrameExternalInputs& external,
    const g1_controller_state& state)
{
    const bool route_mode = external.tuning.mode == G1_TestRoute;
    if (route_mode != (external.route != nullptr)) return false;
    if (!route_mode) return true;
    const int route_index = g1_frame_route_member_index(
        *external.scene, external.route);
    return route_index >= 0 &&
           state.route_index == route_index &&
           !external.route->id.empty() &&
           external.route->waypoints_xz.size() >= 2U &&
           state.route_waypoint >= 1 &&
           static_cast<std::size_t>(state.route_waypoint) <
               external.route->waypoints_xz.size() &&
           state.route_frames >= 0 &&
           deterministic_route_inputs_valid(
               *external.route,
               external.tuning.dt,
               external.tuning.route_speed);
}

static inline bool g1_frame_support_state_equal(
    const support_frame_state& first,
    const support_frame_state& second)
{
    return g1_frame_float_bits_equal(first.height, second.height) &&
           g1_frame_float_bits_equal(first.velocity, second.velocity) &&
           g1_frame_float_bits_equal(
               first.nominal_height, second.nominal_height) &&
           g1_frame_float_bits_equal(
               first.nominal_velocity, second.nominal_velocity) &&
           g1_frame_float_bits_equal(
               first.offset_height, second.offset_height) &&
           g1_frame_float_bits_equal(
               first.offset_velocity, second.offset_velocity) &&
           first.airborne_frames == second.airborne_frames &&
           first.source == second.source &&
           first.initialized == second.initialized;
}

static inline bool g1_frame_support_observation_equal(
    const support_observation& first,
    const support_observation& second)
{
    for (int index = 0; index < 3; ++index) {
        if (!g1_frame_float_bits_equal(
                first.source_height[index],
                second.source_height[index]) ||
            !g1_frame_float_bits_equal(
                first.runtime_height[index],
                second.runtime_height[index]) ||
            !g1_frame_float_bits_equal(
                first.delta[index], second.delta[index])) {
            return false;
        }
    }
    return first.contact[0] == second.contact[0] &&
           first.contact[1] == second.contact[1];
}

static inline bool g1_frame_controller_states_equal(
    const g1_controller_state& first,
    const g1_controller_state& second)
{
    g1_controller_state_memory_range first_ranges[64] = {};
    g1_controller_state_memory_range second_ranges[64] = {};
    int first_count = 0;
    int second_count = 0;
    if (!g1_frame_state_pair_storage_is_exact(
            first,
            second,
            first_ranges,
            first_count,
            second_ranges,
            second_count,
            64)) {
        return false;
    }
    for (int index = 0; index < first_count; ++index) {
        if (first_ranges[index].bytes != second_ranges[index].bytes ||
            std::memcmp(
                first_ranges[index].data,
                second_ranges[index].data,
                first_ranges[index].bytes) != 0) {
            return false;
        }
    }
    return first.frame_index == second.frame_index &&
           first.scene_frame == second.scene_frame &&
           g1_frame_float_bits_equal(
               first.search_time, second.search_time) &&
           g1_frame_float_bits_equal(
               first.search_timer, second.search_timer) &&
           g1_frame_float_bits_equal(
               first.force_search_timer,
               second.force_search_timer) &&
           first.footprint_status == second.footprint_status &&
           g1_frame_footprint_equal(first.footprint, second.footprint) &&
           g1_frame_ik_state_equal(first.ik, second.ik) &&
           g1_frame_ik_result_equal(first.ik_frame, second.ik_frame) &&
           g1_controller_state_pose_clearance_equal(
               first.ik_clearance, second.ik_clearance) &&
           g1_controller_state_pose_clearance_equal(
               first.ik_candidate_clearance,
               second.ik_candidate_clearance) &&
           first.ik_candidate_clearance_status ==
               second.ik_candidate_clearance_status &&
           first.ik_candidate_rejected ==
               second.ik_candidate_rejected &&
           g1_frame_vec3_bits_equal(
               first.transition_src_position,
               second.transition_src_position) &&
           g1_frame_vec3_bits_equal(
               first.transition_dst_position,
               second.transition_dst_position) &&
           g1_frame_quat_bits_equal(
               first.transition_src_rotation,
               second.transition_src_rotation) &&
           g1_frame_quat_bits_equal(
               first.transition_dst_rotation,
               second.transition_dst_rotation) &&
           g1_frame_vec3_bits_equal(
               first.desired_velocity, second.desired_velocity) &&
           g1_frame_vec3_bits_equal(
               first.desired_velocity_change_curr,
               second.desired_velocity_change_curr) &&
           g1_frame_vec3_bits_equal(
               first.desired_velocity_change_prev,
               second.desired_velocity_change_prev) &&
           g1_frame_quat_bits_equal(
               first.desired_rotation, second.desired_rotation) &&
           g1_frame_vec3_bits_equal(
               first.desired_rotation_change_curr,
               second.desired_rotation_change_curr) &&
           g1_frame_vec3_bits_equal(
               first.desired_rotation_change_prev,
               second.desired_rotation_change_prev) &&
           g1_frame_float_bits_equal(
               first.desired_gait, second.desired_gait) &&
           g1_frame_float_bits_equal(
               first.desired_gait_velocity,
               second.desired_gait_velocity) &&
           g1_frame_vec3_bits_equal(
               first.simulation_position,
               second.simulation_position) &&
           g1_frame_vec3_bits_equal(
               first.simulation_velocity,
               second.simulation_velocity) &&
           g1_frame_vec3_bits_equal(
               first.simulation_acceleration,
               second.simulation_acceleration) &&
           g1_frame_quat_bits_equal(
               first.simulation_rotation,
               second.simulation_rotation) &&
           g1_frame_vec3_bits_equal(
               first.simulation_angular_velocity,
               second.simulation_angular_velocity) &&
           g1_frame_command_equal(first.command, second.command) &&
           g1_frame_support_state_equal(first.support, second.support) &&
           g1_frame_support_observation_equal(
               first.support_observation_now,
               second.support_observation_now) &&
           g1_frame_float_bits_equal(
               first.traversal_speed_scale,
               second.traversal_speed_scale) &&
           g1_frame_float_bits_equal(
               first.traversal_speed_scale_velocity,
               second.traversal_speed_scale_velocity) &&
           first.blocked == second.blocked &&
           first.walkability_class == second.walkability_class &&
           g1_frame_float_bits_equal(
               first.blocked_distance, second.blocked_distance) &&
           g1_frame_vec3_bits_equal(
               first.blocked_point, second.blocked_point) &&
           first.route_index == second.route_index &&
           first.route_waypoint == second.route_waypoint &&
           first.route_frames == second.route_frames &&
           g1_frame_float_bits_equal(
               first.camera_azimuth, second.camera_azimuth) &&
           g1_frame_float_bits_equal(
               first.camera_altitude, second.camera_altitude) &&
           g1_frame_float_bits_equal(
               first.camera_distance, second.camera_distance) &&
           first.searched == second.searched &&
           first.transitioned == second.transitioned &&
           g1_frame_float_bits_equal(
               first.incumbent_cost, second.incumbent_cost) &&
           g1_frame_float_bits_equal(
               first.selected_cost, second.selected_cost) &&
           g1_frame_float_bits_equal(
               first.selected_terrain_error,
               second.selected_terrain_error) &&
           g1_frame_float_bits_equal(
               first.adjustment_xz, second.adjustment_xz) &&
           g1_frame_float_bits_equal(
               first.adjustment_y, second.adjustment_y) &&
           g1_frame_float_bits_equal(
               first.clamp_xz, second.clamp_xz) &&
           g1_frame_float_bits_equal(first.clamp_y, second.clamp_y);
}

static inline bool g1_frame_runtime_states_are_logically_equal(
    const G1FrameRuntime& runtime)
{
    const g1_controller_state* states[5] = {};
    g1_frame_runtime_state_pointers(runtime, states);
    for (int state_index = 1; state_index < 5; ++state_index) {
        if (!g1_frame_controller_states_equal(
                *states[0], *states[state_index])) {
            return false;
        }
    }
    return true;
}

static inline bool g1_frame_reset_candidate_is_valid(
    const g1_controller_state& state,
    float initial_search_time)
{
    return g1_controller_state_is_valid(state) &&
           terrain_float_bits(state.search_time) ==
               terrain_float_bits(initial_search_time) &&
           terrain_float_bits(state.search_timer) ==
               terrain_float_bits(initial_search_time) &&
           terrain_float_bits(state.force_search_timer) ==
               terrain_float_bits(initial_search_time) &&
           state.frame_index >= 0 && state.scene_frame == 0 &&
           g1_ik_vec3_is_zero(
               state.command.intent.requested_velocity) &&
           g1_ik_vec3_is_zero(state.command.applied_velocity) &&
           g1_frame_quat_bits_equal(
               state.command.intent.desired_heading,
               state.desired_rotation) &&
           state.footprint_status == G1FootprintOk &&
           !state.footprint.blocked &&
           g1_ik_runtime_state_is_valid(state.ik) &&
           state.ik_candidate_clearance_status == G1ClearanceOk &&
           !state.ik_candidate_rejected &&
           g1_controller_state_pose_clearance_meets_thresholds(
               state.ik_clearance) &&
           g1_controller_state_pose_clearance_meets_thresholds(
               state.ik_candidate_clearance);
}

static inline bool g1_frame_reset_output_preflight(
    const G1FrameRuntime& output,
    const database& db,
    const terrain_support_set& support,
    const scene_pack& scene,
    const G1FrameResetConfig& config,
    char* error,
    int error_capacity)
{
    g1_controller_state_memory_range ranges[5][64] = {};
    int counts[5] = {};
    if (error_capacity < 0 ||
        !g1_frame_runtime_storage_sets_are_safe(
            output, false, ranges, counts) ||
        g1_frame_error_overlaps_object(
            error, error_capacity, &output, sizeof(output)) ||
        g1_frame_error_overlaps_object(
            error, error_capacity, &config, sizeof(config))) {
        return false;
    }
    const g1_controller_state_memory_range output_object = {
        &output, sizeof(output)
    };
    const g1_controller_state_memory_range config_object = {
        &config, sizeof(config)
    };
    if (g1_controller_state_ranges_overlap(
            output_object, config_object) ||
        g1_controller_state_source_storage_overlaps(
            output_object, db, support, scene) ||
        g1_controller_state_source_storage_overlaps(
            config_object, db, support, scene)) {
        return false;
    }
    for (int state_index = 0; state_index < 5; ++state_index) {
        if (g1_frame_error_overlaps_ranges(
                error,
                error_capacity,
                ranges[state_index],
                counts[state_index])) {
            return false;
        }
        for (int index = 0; index < counts[state_index]; ++index) {
            if (g1_controller_state_ranges_overlap(
                    ranges[state_index][index], output_object) ||
                g1_controller_state_ranges_overlap(
                    ranges[state_index][index], config_object) ||
                g1_controller_state_source_storage_overlaps(
                    ranges[state_index][index], db, support, scene)) {
                return false;
            }
        }
    }
    const g1_controller_state_memory_range diagnostic = {
        error,
        error != nullptr && error_capacity > 0
            ? static_cast<std::size_t>(error_capacity)
            : 0U
    };
    if (diagnostic.bytes > 0U &&
        g1_controller_state_source_storage_overlaps(
            diagnostic, db, support, scene)) {
                return false;
    }
    return true;
}

static inline bool g1_frame_intent_is_valid(
    const G1CommandIntent& intent)
{
    return g1_ik_vec3_is_runtime_value(intent.requested_velocity) &&
           ik_quat_is_unit(intent.desired_heading);
}

static inline bool g1_frame_intent_bits_equal(
    const G1CommandIntent& first,
    const G1CommandIntent& second)
{
    return g1_frame_vec3_bits_equal(
               first.requested_velocity,
               second.requested_velocity) &&
           g1_frame_quat_bits_equal(
               first.desired_heading,
               second.desired_heading);
}

static inline bool g1_frame_pose_clearance_is_canonical(
    const G1PoseClearance& value)
{
    const G1PoseClearance canonical;
    return g1_controller_state_pose_clearance_equal(value, canonical);
}

static inline bool g1_frame_footprint_is_canonical(
    const G1FootprintObservation& value)
{
    const G1FootprintObservation canonical;
    return g1_frame_footprint_equal(value, canonical);
}

static inline bool g1_frame_ik_result_is_canonical(
    const G1IkFrameResult& value)
{
    G1IkFrameTransaction transaction;
    transaction.candidate_result = value;
    return g1_ik_runtime_is_disabled_noop(transaction);
}

static inline bool g1_frame_clearance_status_is_known(
    G1ClearanceStatus status)
{
    return status >= G1ClearanceOk &&
           status <= G1ClearanceArithmeticFailure;
}

static inline bool g1_frame_attempted_footprint_is_valid(
    const G1FootprintObservation& footprint)
{
    const G1FootprintBudget budget = g1_footprint_budget();
    if (!g1_ik_runtime_surface_sample_is_valid(
            footprint.root_surface) ||
        footprint.work.sweeps > budget.maximum_sweeps ||
        footprint.work.surface_queries >
            budget.maximum_surface_queries ||
        footprint.work.node_visits > budget.maximum_node_visits ||
        !g1_ik_runtime_walkability_reason_is_valid(
            footprint.blocked_reason) ||
        footprint.blocked !=
            (footprint.blocked_reason != walkability_clear)) {
        return false;
    }
    for (int foot_index = 0; foot_index < 2; ++foot_index) {
        const G1FootprintFootObservation& foot =
            footprint.feet[foot_index];
        if (!g1_ik_runtime_class_is_valid(
                foot.predicted_landing_walkability_class) ||
            !g1_ik_runtime_class_is_valid(
                foot.encountered_walkability_class) ||
            !g1_ik_float_is_runtime_value(
                foot.corridor_minimum_height) ||
            !g1_ik_float_is_runtime_value(
                foot.corridor_maximum_height) ||
            foot.corridor_minimum_height >
                foot.corridor_maximum_height ||
            !terrain_double_is_finite(
                foot.landing_patch_maximum_residual_m) ||
            foot.landing_patch_maximum_residual_m < 0.0 ||
            !terrain_double_is_finite(
                foot.maximum_root_split_m) ||
            foot.maximum_root_split_m < 0.0) {
            return false;
        }
        for (int probe_index = 0; probe_index < 4; ++probe_index) {
            const G1FootprintProbe& probe = foot.probes[probe_index];
            if (!g1_ik_vec3_is_runtime_value(
                    probe.current_sphere_center) ||
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
                return false;
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
                    return false;
                }
            }
        }
        if (!foot.landing_expected) {
            if (foot.landing_patch_ready ||
                foot.landing_sample != UINT32_MAX) {
                return false;
            }
            continue;
        }
        if (foot.current_contact ||
            foot.landing_sample == 0U ||
            foot.landing_sample >=
                G1CommandTrajectorySampleCount ||
            !g1_ik_vec3_is_runtime_value(
                foot.predicted_landing_sole_center) ||
            foot.predicted_landing_surface_status !=
                G1SurfaceQueryValid ||
            !g1_ik_runtime_surface_sample_is_valid(
                foot.predicted_landing_surface)) {
            return false;
        }
        for (int probe_index = 0; probe_index < 4; ++probe_index) {
            if (!g1_ik_runtime_surface_sample_is_valid(
                    foot.probes[probe_index]
                        .selected_landing_surface)) {
                return false;
            }
        }
        if (foot.landing_patch_ready &&
            foot.predicted_landing_walkability_class != 1) {
            return false;
        }
    }
    return true;
}

static inline bool g1_frame_rejected_ik_common_fields_are_valid(
    const G1IkFrameResult& result)
{
    if (result.applied || !result.safe_stop_requested ||
        !g1_ik_runtime_stop_reason_is_valid(result.stop_reason) ||
        result.stop_reason == G1IkStopNone ||
        terrain_float_bits(result.max_correction_radians) != 0U ||
        !g1_root_reach_plan_is_valid(result.root_reach)) {
        return false;
    }
    for (int foot = 0; foot < 2; ++foot) {
        const G1FootFrameResult& staged = result.feet[foot];
        const G1SwingSelectionDiagnostic& selection =
            staged.swing_selection;
        const G1SwingCandidateDiagnostic& selected =
            selection.selected;
        if (!g1_foot_target_is_valid(staged.target) ||
            selection.candidates_evaluated >
                G1SwingLiftCandidateCount ||
            (selection.selected_index != G1SwingNoCandidate &&
             selection.selected_index >=
                 G1SwingLiftCandidateCount) ||
            !g1_frame_clearance_status_is_known(
                selected.clearance_status) ||
            !terrain_double_is_finite(selected.lower_margin_m) ||
            !terrain_double_is_finite(
                selected.witness_upper_margin_m) ||
            !terrain_double_is_finite(
                staged.defensive_swing.lower_margin_m) ||
            !terrain_double_is_finite(
                staged.defensive_swing.witness_upper_m)) {
            return false;
        }
        if (staged.position.applied) {
            if (!g1_ik_leg_result_is_valid(staged.position)) {
                return false;
            }
        } else if (!g1_ik_vec3_is_runtime_value(
                       staged.position.requested_ankle_target) ||
                   !g1_ik_vec3_is_runtime_value(
                       staged.position.clamped_ankle_target) ||
                   !g1_ik_float_is_runtime_value(
                       staged.position.raw_distance_m) ||
                   !g1_ik_float_is_runtime_value(
                       staged.position.clamped_distance_m) ||
                   !g1_ik_float_is_runtime_value(
                       staged.position.max_correction_radians) ||
                   !g1_ik_float_is_runtime_value(
                       staged.position.contact_residual_m)) {
            return false;
        }
        if (!ik_quat_is_unit(
                staged.orientation.target_global_rotation) ||
            !g1_ik_float_is_runtime_value(
                staged.orientation.requested_correction_radians) ||
            !g1_ik_float_is_runtime_value(
                staged.orientation.correction_radians)) {
            return false;
        }
    }
    return true;
}

static inline bool g1_frame_swing_selection_is_canonical(
    const G1SwingSelectionDiagnostic& selection)
{
    return selection.candidates_evaluated == 0U &&
           selection.selected_index == G1SwingNoCandidate &&
           g1_controller_state_swing_candidate_is_canonical(
               selection.selected) &&
           g1_controller_state_clearance_work_is_zero(
               selection.total_clearance_work);
}

static inline bool g1_frame_defensive_swing_is_canonical(
    const G1SwingClearanceValidation& defensive)
{
    return g1_controller_state_double_bits(
               defensive.lower_margin_m) == 0U &&
           g1_controller_state_double_bits(
               defensive.witness_upper_m) == 0U &&
           !defensive.sweep_evaluated &&
           g1_controller_state_clearance_work_is_zero(
               defensive.work);
}

static inline bool g1_frame_leg_result_is_canonical(
    const G1LegSolveResult& position)
{
    const G1LegSolveResult canonical;
    return g1_frame_leg_result_equal(position, canonical);
}

static inline bool g1_frame_orientation_is_canonical(
    const G1FootOrientationResult& orientation)
{
    const G1FootOrientationResult canonical;
    return g1_frame_orientation_equal(orientation, canonical);
}

static inline bool g1_frame_foot_stage_is_canonical(
    const G1FootFrameResult& foot)
{
    return g1_frame_swing_selection_is_canonical(
               foot.swing_selection) &&
           g1_frame_leg_result_is_canonical(foot.position) &&
           g1_frame_orientation_is_canonical(foot.orientation);
}

static inline bool g1_frame_clearance_work_is_at_least(
    const G1ClearanceWork& whole,
    const G1ClearanceWork& part)
{
    return whole.point_queries >= part.point_queries &&
           whole.cells_visited >= part.cells_visited &&
           whole.primitive_triangle_pairs >=
               part.primitive_triangle_pairs &&
           whole.face_patches >= part.face_patches &&
           whole.candidate_tests >= part.candidate_tests &&
           whole.subdivision_nodes >= part.subdivision_nodes;
}

static inline bool g1_frame_clearance_work_is_within_swing_budget(
    const G1ClearanceWork& work,
    uint32_t multiplier)
{
    const G1ClearanceBudget budget =
        g1_swing_foot_clearance_budget();
    return static_cast<uint64_t>(work.point_queries) <=
               static_cast<uint64_t>(
                   budget.maximum_point_queries) * multiplier &&
           static_cast<uint64_t>(work.cells_visited) <=
               static_cast<uint64_t>(
                   budget.maximum_cells) * multiplier &&
           static_cast<uint64_t>(
               work.primitive_triangle_pairs) <=
               static_cast<uint64_t>(
                   budget.maximum_primitive_triangle_pairs) *
                   multiplier &&
           static_cast<uint64_t>(work.face_patches) <=
               static_cast<uint64_t>(
                   budget.maximum_face_patches) * multiplier &&
           static_cast<uint64_t>(work.candidate_tests) <=
               static_cast<uint64_t>(
                   budget.maximum_candidate_tests) * multiplier &&
           static_cast<uint64_t>(work.subdivision_nodes) <=
               static_cast<uint64_t>(
                   budget.maximum_subdivision_nodes) * multiplier;
}

static inline bool g1_frame_leg_result_is_complete(
    const G1LegSolveResult& position,
    const G1LegConfig& config)
{
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
           g1_ik_contact_iterations_have_valid_provenance(position) &&
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
}

static inline bool g1_frame_orientation_is_complete(
    const G1FootOrientationResult& orientation,
    const G1FootTarget& target,
    const G1LegConfig& config)
{
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
}

static inline bool g1_frame_staged_constraints_pass(
    const G1LegSolveResult& position,
    const G1FootOrientationResult& orientation)
{
    return position.reachable &&
           !position.correction_limited &&
           !position.safe_stop_requested &&
           g1_ik_contact_residual_is_converged(
               position.contact_residual_m) &&
           !orientation.correction_limited &&
           !orientation.safe_stop_requested;
}

static inline bool g1_frame_float_word_is_runtime_value(
    uint32_t bits)
{
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return g1_ik_float_is_runtime_value(value);
}

static inline bool g1_frame_selected_swing_is_valid(
    const G1FootFrameResult& foot)
{
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
        !g1_frame_float_word_is_runtime_value(
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
        !g1_frame_clearance_work_is_within_swing_budget(
            selected.clearance_work, 1U) ||
        !g1_frame_clearance_work_is_within_swing_budget(
            selection.total_clearance_work,
            selection.candidates_evaluated) ||
        !g1_frame_clearance_work_is_at_least(
            selection.total_clearance_work,
            selected.clearance_work)) {
        return false;
    }
    float lift = 0.0f;
    std::memcpy(
        &lift, &selected.lift_bits, sizeof(lift));
    float expected_y = 0.0f;
    if (g1_apply_swing_lift_y(
            expected_y,
            foot.target.sole_center.y,
            lift,
            nullptr,
            0) != G1ClearanceOk ||
        g1_ik_runtime_float_bits(expected_y) !=
            selected.materialized_command_y_bits) {
        return false;
    }
    for (int probe = 0; probe < 4; ++probe) {
        for (int axis = 0; axis < 3; ++axis) {
            if (!g1_frame_float_word_is_runtime_value(
                    selected.actual_sphere_center_bits[
                        probe][axis])) {
                return false;
            }
        }
    }
    return true;
}

static inline bool g1_frame_no_swing_is_valid(
    const G1FootFrameResult& foot)
{
    const G1SwingSelectionDiagnostic& selection =
        foot.swing_selection;
    return !foot.recorded_contact &&
           selection.candidates_evaluated ==
               G1SwingLiftCandidateCount &&
           selection.selected_index == G1SwingNoCandidate &&
           g1_controller_state_swing_candidate_is_canonical(
               selection.selected) &&
           g1_frame_clearance_work_is_within_swing_budget(
               selection.total_clearance_work,
               G1SwingLiftCandidateCount) &&
           g1_frame_leg_result_is_canonical(foot.position) &&
           g1_frame_orientation_is_canonical(foot.orientation);
}

static inline bool g1_frame_successful_foot_is_valid(
    const G1FootFrameResult& foot,
    const G1LegConfig& config)
{
    if (!g1_frame_leg_result_is_complete(
            foot.position, config) ||
        !g1_frame_orientation_is_complete(
            foot.orientation, foot.target, config) ||
        !g1_frame_staged_constraints_pass(
            foot.position, foot.orientation)) {
        return false;
    }
    return foot.recorded_contact
        ? g1_frame_swing_selection_is_canonical(
              foot.swing_selection)
        : g1_frame_selected_swing_is_valid(foot);
}

static inline bool g1_frame_rejecting_foot_is_valid(
    const G1FootFrameResult& foot,
    const G1LegConfig& config,
    G1IkStopReason reason)
{
    if (reason == G1IkStopTargetUnreachable) {
        return foot.recorded_contact &&
               g1_frame_swing_selection_is_canonical(
                   foot.swing_selection) &&
               g1_frame_leg_result_is_complete(
                   foot.position, config) &&
               g1_frame_orientation_is_complete(
                   foot.orientation, foot.target, config) &&
               !g1_frame_staged_constraints_pass(
                   foot.position, foot.orientation);
    }
    return reason == G1IkStopNoSwingCandidate &&
           g1_frame_no_swing_is_valid(foot);
}

static inline bool g1_frame_planner_terminal_is_valid(
    const G1IkFrameResult& result,
    const G1LegConfig (&configs)[2])
{
    const bool any_recorded_contact =
        result.feet[0].recorded_contact ||
        result.feet[1].recorded_contact;
    return result.stop_reason == G1IkStopTargetUnreachable &&
           any_recorded_contact &&
           g1_root_reach_plan_is_valid(result.root_reach) &&
           result.root_reach.active &&
           !result.root_reach.common_interval_found &&
           !result.root_reach.applied &&
           terrain_float_bits(
               result.root_reach.root_y_delta_m) == 0U &&
           g1_frame_successful_foot_is_valid(
               result.feet[0], configs[0]) &&
           g1_frame_successful_foot_is_valid(
               result.feet[1], configs[1]);
}

static inline bool g1_frame_rejected_ik_result_is_valid(
    const G1IkFrameResult& result)
{
    if (!g1_frame_rejected_ik_common_fields_are_valid(result)) {
        return false;
    }
    if (result.stop_reason == G1IkStopLandingPatchUnavailable) {
        return g1_frame_foot_stage_is_canonical(result.feet[0]) &&
               g1_frame_foot_stage_is_canonical(result.feet[1]);
    }
    if (result.stop_reason != G1IkStopTargetUnreachable &&
        result.stop_reason != G1IkStopNoSwingCandidate) {
        return false;
    }
    const G1LegConfig configs[2] = {
        g1_left_leg_config(), g1_right_leg_config()
    };
    const bool rejected_after_foot_zero =
        g1_frame_rejecting_foot_is_valid(
            result.feet[0], configs[0], result.stop_reason) &&
        g1_frame_foot_stage_is_canonical(result.feet[1]);
    const bool rejected_after_foot_one =
        g1_frame_successful_foot_is_valid(
            result.feet[0], configs[0]) &&
        g1_frame_rejecting_foot_is_valid(
            result.feet[1], configs[1], result.stop_reason);
    return rejected_after_foot_zero || rejected_after_foot_one ||
           g1_frame_planner_terminal_is_valid(result, configs);
}

static inline bool g1_frame_rejected_ik_common_is_valid(
    const G1IkFrameResult& result,
    const G1FootprintObservation& footprint)
{
    if (!g1_frame_rejected_ik_result_is_valid(result)) {
        return false;
    }
    const bool any_recorded_contact =
        footprint.feet[0].current_contact ||
        footprint.feet[1].current_contact;
    if (result.stop_reason == G1IkStopLandingPatchUnavailable
            ? result.root_reach.active
            : result.root_reach.active != any_recorded_contact) {
        return false;
    }
    for (int foot = 0; foot < 2; ++foot) {
        if (result.feet[foot].recorded_contact !=
                footprint.feet[foot].current_contact ||
            !g1_foot_target_is_valid(
                result.feet[foot].target) ||
            !g1_ik_surface_target_is_valid(
                result.feet[foot].target.surface) ||
            !g1_frame_defensive_swing_is_canonical(
                result.feet[foot].defensive_swing)) {
            return false;
        }
    }
    return true;
}

static inline bool g1_frame_landing_lookahead_is_valid(
    const G1FootTarget& target,
    const G1FootprintFootObservation& footprint,
    const G1LegConfig& config)
{
    return g1_ik_runtime_validate_landing_lookahead(
        target, footprint, config, nullptr, 0);
}

static inline bool g1_frame_landing_patch_rejection_is_valid(
    const G1FrameRejectionDiagnostic& rejection)
{
    const G1FootprintObservation& footprint =
        rejection.attempted_footprint;
    const G1IkFrameResult& result = rejection.ik_frame;
    if (footprint.blocked ||
        !g1_frame_rejected_ik_common_is_valid(
            result, footprint) ||
        !g1_frame_foot_stage_is_canonical(result.feet[0]) ||
        !g1_frame_foot_stage_is_canonical(result.feet[1])) {
        return false;
    }
    int first_unavailable = -1;
    for (int foot = 0; foot < 2; ++foot) {
        const G1FootprintFootObservation& observation =
            footprint.feet[foot];
        if (!observation.current_contact &&
            observation.landing_expected &&
            !observation.landing_patch_ready) {
            first_unavailable = foot;
            break;
        }
    }
    if (first_unavailable < 0) return false;
    const G1LegConfig configs[2] = {
        g1_left_leg_config(), g1_right_leg_config()
    };
    for (int foot = 0; foot < first_unavailable; ++foot) {
        const G1FootprintFootObservation& observation =
            footprint.feet[foot];
        if (!observation.current_contact &&
            observation.landing_expected &&
            observation.landing_patch_ready &&
            !g1_frame_landing_lookahead_is_valid(
                result.feet[foot].target,
                observation,
                configs[foot])) {
            return false;
        }
    }
    return true;
}

static inline bool g1_frame_ik_candidate_rejection_is_valid(
    const G1FrameRejectionDiagnostic& rejection)
{
    const G1FootprintObservation& footprint =
        rejection.attempted_footprint;
    const G1IkFrameResult& result = rejection.ik_frame;
    if (footprint.blocked ||
        !g1_frame_rejected_ik_common_is_valid(
            result, footprint)) {
        return false;
    }
    const G1LegConfig configs[2] = {
        g1_left_leg_config(), g1_right_leg_config()
    };
    for (int foot = 0; foot < 2; ++foot) {
        const G1FootprintFootObservation& observation =
            footprint.feet[foot];
        if (!observation.current_contact &&
            observation.landing_expected &&
            (!observation.landing_patch_ready ||
             !g1_frame_landing_lookahead_is_valid(
                 result.feet[foot].target,
                 observation,
                 configs[foot]))) {
            return false;
        }
    }
    const G1IkStopReason reason = rejection.stop_reason;
    const bool rejected_after_foot_zero =
        g1_frame_rejecting_foot_is_valid(
            result.feet[0], configs[0], reason) &&
        g1_frame_foot_stage_is_canonical(result.feet[1]);
    const bool rejected_after_foot_one =
        g1_frame_successful_foot_is_valid(
            result.feet[0], configs[0]) &&
        g1_frame_rejecting_foot_is_valid(
            result.feet[1], configs[1], reason);
    return rejected_after_foot_zero || rejected_after_foot_one ||
           g1_frame_planner_terminal_is_valid(result, configs);
}

static inline bool g1_frame_rejection_is_canonical(
    const G1FrameRejectionDiagnostic& rejection)
{
    return !rejection.rejected &&
           rejection.stage == G1FrameRejectNone &&
           rejection.stop_reason == G1IkStopNone &&
           !rejection.attempted_footprint_available &&
           rejection.footprint_status == G1FootprintInvalidInput &&
           g1_frame_footprint_is_canonical(
               rejection.attempted_footprint) &&
           !rejection.attempted_ik_available &&
           g1_frame_ik_result_is_canonical(rejection.ik_frame) &&
           !rejection.attempted_pose_available &&
           rejection.pose_status == G1ClearanceInvalidInput &&
           g1_frame_pose_clearance_is_canonical(
               rejection.pose_clearance);
}

static inline bool g1_frame_rejection_is_valid(
    const G1FrameRejectionDiagnostic& rejection)
{
    if (!rejection.rejected) {
        return g1_frame_rejection_is_canonical(rejection);
    }
    if (rejection.stage <= G1FrameRejectNone ||
        rejection.stage > G1FrameRejectPoseCertificate ||
        !g1_ik_runtime_stop_reason_is_valid(rejection.stop_reason) ||
        rejection.stop_reason == G1IkStopNone ||
        rejection.footprint_status < G1FootprintOk ||
        rejection.footprint_status >
            G1FootprintArithmeticFailure ||
        !g1_frame_clearance_status_is_known(
            rejection.pose_status)) {
        return false;
    }
    if (rejection.attempted_footprint_available) {
        if (rejection.footprint_status != G1FootprintOk ||
            !g1_frame_attempted_footprint_is_valid(
                rejection.attempted_footprint)) {
            return false;
        }
    } else if ((rejection.footprint_status !=
                    G1FootprintOutsideDomain &&
                rejection.footprint_status !=
                    G1FootprintBudgetExceeded) ||
               !g1_frame_footprint_is_canonical(
                   rejection.attempted_footprint)) {
        return false;
    }
    if (rejection.attempted_ik_available) {
        if (!g1_frame_rejected_ik_result_is_valid(
                rejection.ik_frame) ||
            rejection.ik_frame.stop_reason !=
                rejection.stop_reason) {
            return false;
        }
    } else if (!g1_frame_ik_result_is_canonical(
                   rejection.ik_frame)) {
        return false;
    }
    if (rejection.attempted_pose_available) {
        if (rejection.pose_status != G1ClearanceOk ||
            !g1_controller_state_pose_clearance_is_coherent(
                rejection.pose_clearance) ||
            g1_controller_state_pose_clearance_meets_thresholds(
                rejection.pose_clearance)) {
            return false;
        }
    } else if (!g1_frame_pose_clearance_is_canonical(
                   rejection.pose_clearance)) {
        return false;
    }

    switch (rejection.stage) {
    case G1FrameRejectFootprint:
        return !rejection.attempted_ik_available &&
               !rejection.attempted_pose_available &&
               rejection.pose_status == G1ClearanceInvalidInput &&
               ((rejection.stop_reason ==
                     G1IkStopFootprintOutsideDomain &&
                 !rejection.attempted_footprint_available &&
                 rejection.footprint_status ==
                     G1FootprintOutsideDomain) ||
                (rejection.stop_reason ==
                     G1IkStopFootprintBudgetExceeded &&
                 !rejection.attempted_footprint_available &&
                 rejection.footprint_status ==
                     G1FootprintBudgetExceeded) ||
                (rejection.stop_reason ==
                     G1IkStopFootprintBlocked &&
                 rejection.attempted_footprint_available &&
                 rejection.attempted_footprint.blocked));
    case G1FrameRejectLandingPatch:
        return rejection.stop_reason ==
                   G1IkStopLandingPatchUnavailable &&
               rejection.attempted_footprint_available &&
               rejection.attempted_ik_available &&
               !rejection.attempted_pose_available &&
               rejection.pose_status == G1ClearanceInvalidInput &&
               g1_frame_landing_patch_rejection_is_valid(
                   rejection);
    case G1FrameRejectIkCandidate:
        return (rejection.stop_reason ==
                    G1IkStopTargetUnreachable ||
                rejection.stop_reason ==
                    G1IkStopNoSwingCandidate) &&
               rejection.attempted_footprint_available &&
               rejection.attempted_ik_available &&
               !rejection.attempted_pose_available &&
               rejection.pose_status == G1ClearanceInvalidInput &&
               g1_frame_ik_candidate_rejection_is_valid(
                   rejection);
    case G1FrameRejectPoseCertificate:
        return rejection.stop_reason ==
                   G1IkStopPoseClearanceRejected &&
               rejection.attempted_footprint_available &&
               !rejection.attempted_ik_available &&
               rejection.pose_status >= G1ClearanceOk &&
               rejection.pose_status <= G1ClearanceUncertified &&
               (rejection.attempted_pose_available ||
                rejection.pose_status != G1ClearanceOk);
    default:
        return false;
    }
}

static inline bool g1_frame_publication_is_valid(
    const G1FramePublication& publication)
{
    return g1_frame_intent_is_valid(publication.requested_intent) &&
           publication.presentation_frame >= 0 &&
           g1_frame_rejection_is_valid(publication.rejection) &&
           publication.ik_safe_stop_latched ==
               publication.rejection.rejected;
}

static inline bool g1_frame_pose_diagnostic_equal(
    const motion_match_pose_diagnostic& first,
    const motion_match_pose_diagnostic& second)
{
    return g1_frame_float_bits_equal(first.hips_y, second.hips_y) &&
           g1_frame_float_bits_equal(
               first.hips_clearance, second.hips_clearance) &&
           g1_frame_float_bits_equal(
               first.left_toe_clearance,
               second.left_toe_clearance) &&
           g1_frame_float_bits_equal(
               first.right_toe_clearance,
               second.right_toe_clearance) &&
           g1_frame_float_bits_equal(
               first.minimum_clearance,
               second.minimum_clearance);
}

static inline bool g1_frame_accepted_diagnostic_equal(
    const G1FrameAcceptedDiagnostic& first,
    const G1FrameAcceptedDiagnostic& second)
{
    if (first.ready != second.ready ||
        first.presentation_frame != second.presentation_frame ||
        first.scene_frame != second.scene_frame ||
        !g1_frame_vec3_bits_equal(
            first.route.command, second.route.command) ||
        first.route.waypoint != second.route.waypoint ||
        first.route.complete != second.route.complete ||
        first.traversal.blocked != second.traversal.blocked ||
        first.traversal.walkability_class !=
            second.traversal.walkability_class ||
        first.traversal.reason != second.traversal.reason ||
        !g1_frame_float_bits_equal(
            first.traversal.distance,
            second.traversal.distance) ||
        !g1_frame_float_bits_equal(
            first.traversal.commanded_speed,
            second.traversal.commanded_speed) ||
        !g1_frame_float_bits_equal(
            first.traversal.applied_speed,
            second.traversal.applied_speed) ||
        !g1_frame_vec3_bits_equal(
            first.traversal.point,
            second.traversal.point) ||
        first.query_database_frame != second.query_database_frame ||
        first.query_range != second.query_range ||
        first.selected_database_frame !=
            second.selected_database_frame ||
        !g1_frame_pose_diagnostic_equal(
            first.raw_selected, second.raw_selected) ||
        !g1_frame_pose_diagnostic_equal(
            first.inertialized, second.inertialized) ||
        !g1_frame_pose_diagnostic_equal(
            first.support_retargeted,
            second.support_retargeted) ||
        !g1_frame_pose_diagnostic_equal(
            first.rendered, second.rendered) ||
        first.matching_enabled != second.matching_enabled ||
        first.adjustment_enabled != second.adjustment_enabled ||
        first.clamping_enabled != second.clamping_enabled ||
        first.ik_enabled != second.ik_enabled ||
        !g1_frame_float_bits_equal(
            first.effective_terrain_weight,
            second.effective_terrain_weight)) {
        return false;
    }
    for (int index = 0; index < 31; ++index) {
        if (!g1_frame_float_bits_equal(
                first.query[index], second.query[index])) {
            return false;
        }
    }
    for (int index = 0; index < 4; ++index) {
        if (!g1_frame_float_bits_equal(
                first.terrain_query.values[index],
                second.terrain_query.values[index]) ||
            !g1_frame_vec3_bits_equal(
                first.terrain_query.points[index],
                second.terrain_query.points[index])) {
            return false;
        }
    }
    return true;
}

static inline bool g1_frame_pose_diagnostic_is_valid(
    const motion_match_pose_diagnostic& value)
{
    return terrain_float_is_normal_or_zero_query(value.hips_y) &&
           terrain_float_is_normal_or_zero_query(
               value.hips_clearance) &&
           terrain_float_is_normal_or_zero_query(
               value.left_toe_clearance) &&
           terrain_float_is_normal_or_zero_query(
               value.right_toe_clearance) &&
           terrain_float_is_normal_or_zero_query(
               value.minimum_clearance);
}

static inline bool g1_frame_accepted_diagnostic_is_valid(
    const G1FrameAcceptedDiagnostic& diagnostic)
{
    if (!diagnostic.ready) {
        const G1FrameAcceptedDiagnostic canonical;
        return g1_frame_accepted_diagnostic_equal(
            diagnostic, canonical);
    }
    if (diagnostic.presentation_frame < 0 ||
        diagnostic.scene_frame < 0 ||
        !g1_ik_vec3_is_runtime_value(diagnostic.route.command) ||
        diagnostic.route.waypoint < 0 ||
        !g1_ik_runtime_walkability_reason_is_valid(
            diagnostic.traversal.reason) ||
        !g1_ik_runtime_class_is_valid(
            diagnostic.traversal.walkability_class) ||
        diagnostic.traversal.blocked !=
            (diagnostic.traversal.reason != walkability_clear) ||
        !g1_frame_nonnegative_runtime_value(
            diagnostic.traversal.distance) ||
        !g1_frame_nonnegative_runtime_value(
            diagnostic.traversal.commanded_speed) ||
        !g1_frame_nonnegative_runtime_value(
            diagnostic.traversal.applied_speed) ||
        !g1_ik_vec3_is_runtime_value(
            diagnostic.traversal.point) ||
        diagnostic.query_database_frame < 0 ||
        diagnostic.query_range < 0 ||
        diagnostic.selected_database_frame < 0 ||
        !g1_frame_pose_diagnostic_is_valid(
            diagnostic.raw_selected) ||
        !g1_frame_pose_diagnostic_is_valid(
            diagnostic.inertialized) ||
        !g1_frame_pose_diagnostic_is_valid(
            diagnostic.support_retargeted) ||
        !g1_frame_pose_diagnostic_is_valid(
            diagnostic.rendered) ||
        !g1_frame_nonnegative_runtime_value(
            diagnostic.effective_terrain_weight) ||
        diagnostic.effective_terrain_weight > 10.0f) {
        return false;
    }
    for (int dimension = 0; dimension < 31; ++dimension) {
        if (!terrain_float_is_normal_or_zero_query(
                diagnostic.query[dimension])) {
            return false;
        }
    }
    for (int sample = 0; sample < 4; ++sample) {
        if (!terrain_float_is_normal_or_zero_query(
                diagnostic.terrain_query.values[sample]) ||
            !g1_ik_vec3_is_runtime_value(
                diagnostic.terrain_query.points[sample])) {
            return false;
        }
    }
    return true;
}

static inline bool g1_frame_accepted_diagnostic_matches_success(
    const G1FrameAcceptedDiagnostic& diagnostic,
    const g1_controller_state& working_state,
    const G1FrameExternalInputs& external)
{
    if (external.db == nullptr ||
        working_state.scene_frame <= 0 ||
        working_state.global_bone_positions.size != G1_BoneCount ||
        working_state.ik_global_bone_positions.size != G1_BoneCount ||
        diagnostic.scene_frame != working_state.scene_frame - 1 ||
        diagnostic.matching_enabled !=
            (external.tuning.mode != G1_TestSequential) ||
        diagnostic.adjustment_enabled !=
            external.tuning.adjustment_enabled ||
        diagnostic.clamping_enabled !=
            external.tuning.clamping_enabled ||
        diagnostic.ik_enabled != external.tuning.ik_enabled ||
        !g1_frame_float_bits_equal(
            diagnostic.effective_terrain_weight,
            external.tuning.effective_terrain_weight) ||
        !g1_frame_float_bits_equal(
            diagnostic.support_retargeted.hips_y,
            working_state.global_bone_positions(G1_Hips).y) ||
        !g1_frame_float_bits_equal(
            diagnostic.rendered.hips_y,
            working_state.ik_global_bone_positions(G1_Hips).y)) {
        return false;
    }
    const database& db = *external.db;
    const int frames = db.nframes();
    if (frames <= 0 ||
        diagnostic.query_database_frame < 0 ||
        diagnostic.query_database_frame >= frames ||
        diagnostic.selected_database_frame < 0 ||
        diagnostic.selected_database_frame >= frames ||
        diagnostic.query_range < 0 ||
        diagnostic.query_range >= db.range_starts.size ||
        diagnostic.query_range >= db.range_stops.size) {
        return false;
    }
    return diagnostic.query_database_frame >=
               db.range_starts(diagnostic.query_range) &&
           diagnostic.query_database_frame <
               db.range_stops(diagnostic.query_range);
}

static inline bool g1_frame_runtime_observation_relation_is_valid(
    const G1FrameRuntime& runtime)
{
    const G1FramePublication& publication = runtime.publication;
    const G1FrameAcceptedDiagnostic& diagnostic =
        runtime.accepted_diagnostic;
    if (!diagnostic.ready &&
        !publication.rejection.rejected &&
        !g1_frame_intent_bits_equal(
            publication.requested_intent,
            runtime.accepted_state.command.intent)) {
        return false;
    }
    if (!diagnostic.ready) return true;
    if (runtime.accepted_state.scene_frame <= 0 ||
        diagnostic.scene_frame !=
            runtime.accepted_state.scene_frame - 1) {
        return false;
    }
    return publication.rejection.rejected ||
           diagnostic.presentation_frame ==
               publication.presentation_frame;
}

static inline bool g1_frame_ranges_overlap_object(
    const g1_controller_state_memory_range* ranges,
    int count,
    const void* object,
    std::size_t object_bytes)
{
    if (ranges == nullptr || count < 0) return true;
    const g1_controller_state_memory_range object_range = {
        object, object_bytes
    };
    for (int index = 0; index < count; ++index) {
        if (g1_controller_state_ranges_overlap(
                ranges[index], object_range)) {
            return true;
        }
    }
    return false;
}

static inline bool g1_frame_ranges_overlap_sources(
    const g1_controller_state_memory_range* ranges,
    int count,
    const database& db,
    const terrain_support_set& support,
    const scene_pack& scene)
{
    if (ranges == nullptr || count < 0) return true;
    for (int index = 0; index < count; ++index) {
        if (g1_controller_state_source_storage_overlaps(
                ranges[index], db, support, scene)) {
            return true;
        }
    }
    return false;
}

static inline bool g1_frame_transaction_preflight(
    const G1FrameRuntime& runtime,
    const G1FrameExternalInputs& external,
#if defined(G1_FRAME_TRANSACTION_ENABLE_TEST_SEAM)
    const G1FrameTransactionTestSeam* test_seam,
#endif
    char* error,
    int error_capacity)
{
    g1_controller_state_memory_range ranges[5][64] = {};
    int counts[5] = {};
    if (error_capacity < 0 ||
        external.db == nullptr ||
        external.support == nullptr ||
        external.scene == nullptr ||
        !g1_frame_runtime_storage_sets_are_safe(
            runtime, true, ranges, counts) ||
        g1_frame_error_overlaps_object(
            error, error_capacity, &runtime, sizeof(runtime)) ||
        g1_frame_error_overlaps_object(
            error, error_capacity, &external, sizeof(external))) {
        return false;
    }

    const g1_controller_state_memory_range runtime_object = {
        &runtime, sizeof(runtime)
    };
    const g1_controller_state_memory_range external_object = {
        &external, sizeof(external)
    };
    const g1_controller_state_memory_range diagnostic = {
        error,
        error != nullptr && error_capacity > 0
            ? static_cast<std::size_t>(error_capacity)
            : 0U
    };
    if (g1_controller_state_ranges_overlap(
            runtime_object, external_object) ||
        g1_controller_state_source_storage_overlaps(
            runtime_object,
            *external.db,
            *external.support,
            *external.scene) ||
        g1_controller_state_source_storage_overlaps(
            external_object,
            *external.db,
            *external.support,
            *external.scene) ||
        (diagnostic.bytes > 0U &&
         g1_controller_state_source_storage_overlaps(
             diagnostic,
             *external.db,
             *external.support,
             *external.scene))) {
        return false;
    }

#if defined(G1_FRAME_TRANSACTION_ENABLE_TEST_SEAM)
    if (test_seam != nullptr &&
        reinterpret_cast<uintptr_t>(test_seam) %
                alignof(G1FrameTransactionTestSeam) !=
            0U) {
        return false;
    }
    if (g1_frame_error_overlaps_object(
            error,
            error_capacity,
            test_seam,
            test_seam != nullptr ? sizeof(*test_seam) : 0U) ||
        (test_seam != nullptr &&
         (test_seam->control.injected_stage <
              G1FrameStageInputRouteCommand ||
          test_seam->control.injected_stage > G1FrameStageCount ||
          test_seam->control.injected_outcome <
              G1FrameInjectContinue ||
          test_seam->control.injected_outcome >
              G1FrameInjectGlobalError))) {
        return false;
    }
    const G1CandidateCertificationTrace* certification_trace =
        test_seam != nullptr
            ? test_seam->certification_trace
            : nullptr;
    if (certification_trace != nullptr &&
        reinterpret_cast<uintptr_t>(certification_trace) %
                alignof(G1CandidateCertificationTrace) !=
            0U) {
        return false;
    }
    if (test_seam != nullptr &&
        (g1_ik_memory_ranges_overlap(
             &runtime,
             sizeof(runtime),
             test_seam,
             sizeof(*test_seam)) ||
         g1_ik_memory_ranges_overlap(
             &external,
             sizeof(external),
             test_seam,
             sizeof(*test_seam)) ||
         g1_controller_state_source_storage_overlaps(
             {test_seam, sizeof(*test_seam)},
             *external.db,
             *external.support,
             *external.scene))) {
        return false;
    }
    if (certification_trace != nullptr &&
        (g1_ik_memory_ranges_overlap(
             &runtime,
             sizeof(runtime),
             certification_trace,
             sizeof(*certification_trace)) ||
         g1_ik_memory_ranges_overlap(
             &external,
             sizeof(external),
             certification_trace,
             sizeof(*certification_trace)) ||
         g1_ik_memory_ranges_overlap(
             test_seam,
             sizeof(*test_seam),
             certification_trace,
             sizeof(*certification_trace)) ||
         g1_frame_error_overlaps_object(
             error,
             error_capacity,
             certification_trace,
             sizeof(*certification_trace)) ||
         g1_controller_state_source_storage_overlaps(
             {certification_trace, sizeof(*certification_trace)},
             *external.db,
             *external.support,
             *external.scene))) {
        return false;
    }
#endif

    for (int state_index = 0; state_index < 5; ++state_index) {
        if (g1_frame_error_overlaps_ranges(
                error,
                error_capacity,
                ranges[state_index],
                counts[state_index]) ||
            g1_frame_ranges_overlap_object(
                ranges[state_index],
                counts[state_index],
                &runtime,
                sizeof(runtime)) ||
            g1_frame_ranges_overlap_object(
                ranges[state_index],
                counts[state_index],
                &external,
                sizeof(external)) ||
            g1_frame_ranges_overlap_sources(
                ranges[state_index],
                counts[state_index],
                *external.db,
                *external.support,
                *external.scene)) {
            return false;
        }
#if defined(G1_FRAME_TRANSACTION_ENABLE_TEST_SEAM)
        if (test_seam != nullptr &&
            g1_frame_ranges_overlap_object(
                ranges[state_index],
                counts[state_index],
                test_seam,
                sizeof(*test_seam))) {
            return false;
        }
        if (certification_trace != nullptr &&
            g1_frame_ranges_overlap_object(
                ranges[state_index],
                counts[state_index],
                certification_trace,
                sizeof(*certification_trace))) {
            return false;
        }
#endif
    }

    return g1_frame_artifacts_are_valid(
               *external.db,
               *external.support,
               *external.scene,
               error,
               error_capacity) &&
           g1_frame_input_snapshot_is_valid(
               external.input, external.heading_override) &&
           g1_frame_tuning_is_valid(external.tuning) &&
           g1_controller_state_is_valid(runtime.accepted_state) &&
           g1_frame_publication_is_valid(runtime.publication) &&
           g1_frame_accepted_diagnostic_is_valid(
               runtime.accepted_diagnostic) &&
           g1_frame_runtime_observation_relation_is_valid(runtime) &&
           g1_frame_float_bits_equal(
               external.tuning.initial_search_time,
               runtime.accepted_state.search_time) &&
           g1_frame_route_relation_is_valid(
               external, runtime.accepted_state);
}
static inline bool g1_frame_rejection_matches_scratch_transaction(
    const G1FrameRejectionDiagnostic& rejection,
    const G1FrameTransactionScratch& scratch)
{
    if (rejection.footprint_status != scratch.footprint_status ||
        !g1_frame_footprint_equal(
            rejection.attempted_footprint,
            scratch.footprint)) {
        return false;
    }

    if (rejection.attempted_ik_available) {
        if (scratch.rejection_branch != G1FrameCertificateIk) {
            return false;
        }
        const G1IkFrameTransaction& transaction =
            scratch.ik_certificate.ik_transaction;
        G1IkRejectionCheckpoint checkpoint =
            G1IkRejectionAfterBegin;
        if (transaction.next_foot == 0U) {
            if (rejection.stage != G1FrameRejectLandingPatch) {
                return false;
            }
        } else if (transaction.next_foot == 1U) {
            if (rejection.stage != G1FrameRejectIkCandidate) {
                return false;
            }
            checkpoint = G1IkRejectionAfterFoot0;
        } else if (transaction.next_foot == 2U) {
            if (rejection.stage != G1FrameRejectIkCandidate) {
                return false;
            }
            checkpoint = G1IkRejectionAfterFoot1;
        } else {
            return false;
        }

        G1IkFrameResult derived;
        if (!g1_ik_frame_rejection_snapshot(
                derived,
                transaction,
                checkpoint,
                nullptr,
                0) ||
            !g1_frame_ik_result_equal(
                derived, rejection.ik_frame)) {
            return false;
        }
    }

    if (rejection.stage == G1FrameRejectPoseCertificate) {
        const G1FrameBranchCertificateScratch* certificate = nullptr;
        if (scratch.rejection_branch == G1FrameCertificateRaw) {
            certificate = &scratch.raw_certificate;
        } else if (scratch.rejection_branch == G1FrameCertificateIk) {
            certificate = &scratch.ik_certificate;
        } else {
            return false;
        }
        if (rejection.pose_status != certificate->pose_status ||
            !g1_controller_state_pose_clearance_equal(
                rejection.pose_clearance,
                certificate->pose_clearance)) {
            return false;
        }
    } else if (rejection.attempted_pose_available) {
        return false;
    } else if (!rejection.attempted_ik_available &&
               scratch.rejection_branch != G1FrameCertificateNone) {
        return false;
    }
    return true;
}

static inline bool g1_frame_publish_finite_rejection(
    G1FrameRuntime& runtime,
    const G1FrameTransactionScratch& scratch,
    const G1FrameExternalInputs& external,
    bool authenticate_scratch_transaction)
{
    if (!scratch.requested_intent_ready ||
        !g1_frame_intent_is_valid(scratch.requested_intent) ||
        !g1_frame_rejection_is_valid(scratch.rejection) ||
        !scratch.rejection.rejected ||
        (authenticate_scratch_transaction &&
         !g1_frame_rejection_matches_scratch_transaction(
             scratch.rejection,
             scratch))) {
        return false;
    }
    G1FramePublication candidate = runtime.publication;
    candidate.requested_intent = scratch.requested_intent;
    candidate.rejection = scratch.rejection;
    candidate.ik_safe_stop_latched = true;
    candidate.presentation_frame =
        external.input.presentation_frame;
    if (!g1_frame_publication_is_valid(candidate)) return false;
    runtime.publication = candidate;
    return true;
}

static inline bool g1_frame_success_iteration_provenance_is_valid(
    const G1IkFrameResult& result)
{
    if (!result.applied) return true;
    for (int foot = 0; foot < 2; ++foot) {
        if (!g1_ik_contact_iterations_have_valid_provenance(
                result.feet[foot].position)) {
            return false;
        }
    }
    return true;
}

static inline bool g1_frame_success_ik_matches_producer(
    const G1IkFrameResult& result,
    const G1IkFrameTransaction& transaction)
{
    return transaction.initialized &&
           transaction.next_foot == 2U &&
           g1_ik_runtime_iteration_transcript_matches(transaction) &&
           g1_frame_ik_result_equal(
               result, transaction.candidate_result);
}

static inline bool g1_frame_public_pose_arrays_equal(
    const g1_controller_state& first,
    const g1_controller_state& second)
{
    return g1_frame_array_values_equal(
               first.ik_bone_positions,
               second.ik_bone_positions) &&
           g1_frame_array_values_equal(
               first.ik_bone_rotations,
               second.ik_bone_rotations) &&
           g1_frame_array_values_equal(
               first.ik_global_bone_positions,
               second.ik_global_bone_positions) &&
           g1_frame_array_values_equal(
               first.ik_global_bone_rotations,
               second.ik_global_bone_rotations) &&
           g1_frame_array_values_equal(
               first.ik_candidate_bone_positions,
               second.ik_candidate_bone_positions) &&
           g1_frame_array_values_equal(
               first.ik_candidate_bone_rotations,
               second.ik_candidate_bone_rotations) &&
           g1_frame_array_values_equal(
               first.ik_candidate_global_bone_positions,
               second.ik_candidate_global_bone_positions) &&
           g1_frame_array_values_equal(
               first.ik_candidate_global_bone_rotations,
               second.ik_candidate_global_bone_rotations);
}

static inline bool g1_frame_branch_certificate_matches_state(
    const g1_controller_state& state,
    const G1FrameBranchCertificateScratch& certificate);

static inline bool g1_frame_success_candidates_are_valid(
    const g1_controller_state& working_state,
    const g1_controller_state& raw_state,
    const g1_controller_state& ik_state,
    const G1FrameTransactionScratch& scratch,
    const G1FrameExternalInputs& external,
    G1FramePublication& publication,
    G1FrameAcceptedDiagnostic& accepted_diagnostic)
{
    const G1FrameBranchCertificateScratch& visible_certificate =
        external.tuning.ik_enabled
            ? scratch.ik_certificate
            : scratch.raw_certificate;
    const g1_controller_state& visible_state =
        external.tuning.ik_enabled ? ik_state : raw_state;
    if (!g1_controller_state_is_valid(working_state) ||
        !g1_frame_branch_certificate_matches_state(
            raw_state, scratch.raw_certificate) ||
        !g1_frame_branch_certificate_matches_state(
            ik_state, scratch.ik_certificate) ||
        !g1_frame_success_iteration_provenance_is_valid(
            scratch.ik_certificate.ik_transaction.candidate_result) ||
        !g1_frame_success_ik_matches_producer(
            scratch.ik_certificate.ik_transaction.candidate_result,
            scratch.ik_certificate.ik_transaction) ||
        !g1_frame_success_iteration_provenance_is_valid(
            working_state.ik_frame) ||
        !g1_frame_success_ik_matches_producer(
            working_state.ik_frame,
            visible_certificate.ik_transaction) ||
        !g1_frame_ik_state_equal(
            working_state.ik,
            scratch.ik_certificate.ik_transaction.candidate_state) ||
        (!external.tuning.ik_enabled &&
         !g1_frame_ik_result_is_canonical(
             working_state.ik_frame)) ||
        !g1_frame_public_pose_arrays_equal(
            working_state, visible_state) ||
        visible_certificate.pose_status != G1ClearanceOk ||
        !g1_controller_state_pose_clearance_equal(
            working_state.ik_clearance,
            visible_certificate.pose_clearance) ||
        !g1_frame_route_relation_is_valid(external, working_state) ||
        !g1_frame_float_bits_equal(
            external.tuning.initial_search_time,
            working_state.search_time) ||
        !scratch.requested_intent_ready ||
        !g1_frame_intent_is_valid(scratch.requested_intent) ||
        !scratch.accepted_diagnostic_ready ||
        !scratch.accepted_diagnostic_candidate.ready ||
        !g1_frame_accepted_diagnostic_is_valid(
            scratch.accepted_diagnostic_candidate) ||
        !g1_frame_accepted_diagnostic_matches_success(
            scratch.accepted_diagnostic_candidate,
            working_state,
            external) ||
        scratch.accepted_diagnostic_candidate.presentation_frame !=
            external.input.presentation_frame) {
        return false;
    }
    G1FramePublication publication_candidate;
    publication_candidate.requested_intent =
        scratch.requested_intent;
    publication_candidate.presentation_frame =
        external.input.presentation_frame;
    if (!g1_frame_publication_is_valid(publication_candidate)) {
        return false;
    }
    publication = publication_candidate;
    accepted_diagnostic =
        scratch.accepted_diagnostic_candidate;
    return true;
}

enum G1CandidateEvaluationOutcome : uint32_t
{
    G1CandidateDualAccepted = 0U,
    G1CandidateFiniteRejected,
    G1CandidateGlobalError,
};

#if defined(G1_FRAME_TRANSACTION_ENABLE_TEST_SEAM)
static inline bool g1_frame_candidate_dual_outcome_is_valid(
    G1CandidateDisposition common,
    G1CandidateDisposition raw,
    G1CandidateDisposition ik)
{
    return common == G1CandidateDispositionAccepted &&
           raw == G1CandidateDispositionAccepted &&
           ik == G1CandidateDispositionAccepted;
}
#endif

static inline G1FrameStageOutcome g1_frame_run_one_stage(
    G1FrameTransactionStage stage,
    G1FrameStageRunner run_stage,
    g1_controller_state& state,
    G1FrameTransactionScratch& scratch,
    const G1FrameExternalInputs& external,
#if defined(G1_FRAME_TRANSACTION_ENABLE_TEST_SEAM)
    const G1FrameTransactionTestSeam* test_seam,
#endif
    char* error,
    int error_capacity)
{
    if (run_stage == nullptr) return G1FrameStageGlobalError;
    const G1FrameStageOutcome outcome = run_stage(
        stage, state, scratch, external, error, error_capacity);
    if (outcome != G1FrameStageContinue) {
        return outcome == G1FrameStageFiniteReject ||
               outcome == G1FrameStageGlobalError
            ? outcome
            : G1FrameStageGlobalError;
    }
#if defined(G1_FRAME_TRANSACTION_ENABLE_TEST_SEAM)
    if (test_seam != nullptr && test_seam->hook != nullptr) {
        const G1FrameInjectedOutcome injected = test_seam->hook(
            stage,
            state,
            scratch,
            external,
            test_seam->control,
            error,
            error_capacity);
        if (injected == G1FrameInjectFiniteReject) {
            return G1FrameStageFiniteReject;
        }
        if (injected == G1FrameInjectGlobalError) {
            return G1FrameStageGlobalError;
        }
        if (injected != G1FrameInjectContinue) {
            return G1FrameStageGlobalError;
        }
    }
#endif
    return G1FrameStageContinue;
}

static inline bool g1_frame_branch_certificate_matches_state(
    const g1_controller_state& state,
    const G1FrameBranchCertificateScratch& certificate)
{
    return g1_controller_state_is_valid(state) &&
           g1_frame_success_iteration_provenance_is_valid(
               state.ik_frame) &&
           g1_frame_success_ik_matches_producer(
               state.ik_frame, certificate.ik_transaction) &&
           g1_frame_ik_state_equal(
               state.ik,
               certificate.ik_transaction.candidate_state) &&
           certificate.pose_status == G1ClearanceOk &&
           g1_controller_state_pose_clearance_equal(
               state.ik_clearance, certificate.pose_clearance);
}

#if defined(G1_FRAME_TRANSACTION_ENABLE_TEST_SEAM)
static inline void g1_frame_candidate_trace_failure(
    G1CandidateAttemptTraceRecord* trace_record,
    const G1FrameTransactionScratch& scratch)
{
    if (trace_record == nullptr) return;
    trace_record->rejection_stage = scratch.rejection.stage;
    trace_record->stop_reason = scratch.rejection.stop_reason;
}
#endif

static inline G1CandidateEvaluationOutcome g1_frame_candidate_evaluate(
    G1FrameCandidateWorkspace& workspace,
    G1FrameTransactionScratch& candidate_scratch,
    const g1_controller_state& immutable_baseline,
    const G1FrameTransactionScratch& prefix_scratch,
    const G1CandidateRecord& record,
    G1FrameStageRunner run_stage,
    const G1FrameExternalInputs& external,
#if defined(G1_FRAME_TRANSACTION_ENABLE_TEST_SEAM)
    G1CandidateAttemptTraceRecord* trace_record,
    const G1FrameTransactionTestSeam* test_seam,
#endif
    char* error,
    int error_capacity)
{
#if defined(G1_FRAME_TRANSACTION_ENABLE_TEST_SEAM)
    G1CandidateCertificationTrace* certification_trace =
        test_seam != nullptr
            ? test_seam->certification_trace
            : nullptr;
#endif
    if (!g1_controller_state_copy(
            workspace.common_state,
            immutable_baseline,
            error,
            error_capacity)) {
#if defined(G1_FRAME_TRANSACTION_ENABLE_TEST_SEAM)
        if (trace_record != nullptr) {
            trace_record->common = G1CandidateDispositionGlobalError;
        }
#endif
        return G1CandidateGlobalError;
    }
    candidate_scratch = prefix_scratch;
    candidate_scratch.active_candidate = record;
    candidate_scratch.raw_certificate =
        G1FrameBranchCertificateScratch{};
    candidate_scratch.ik_certificate =
        G1FrameBranchCertificateScratch{};
    candidate_scratch.rejection_branch = G1FrameCertificateNone;
    candidate_scratch.rejection = G1FrameRejectionDiagnostic{};
    candidate_scratch.accepted_diagnostic_candidate =
        G1FrameAcceptedDiagnostic{};
    candidate_scratch.accepted_diagnostic_ready = false;

#if defined(G1_FRAME_TRANSACTION_ENABLE_TEST_SEAM)
    if (certification_trace != nullptr) {
        if (certification_trace->common_evaluations == UINT32_MAX) {
            return G1CandidateGlobalError;
        }
        ++certification_trace->common_evaluations;
    }
#endif
    for (int stage_index = G1FrameStageCandidateApply;
         stage_index <= G1FrameStageFootprintObservation;
         ++stage_index) {
        const G1FrameStageOutcome outcome = g1_frame_run_one_stage(
            static_cast<G1FrameTransactionStage>(stage_index),
            run_stage,
            workspace.common_state,
            candidate_scratch,
            external,
#if defined(G1_FRAME_TRANSACTION_ENABLE_TEST_SEAM)
            test_seam,
#endif
            error,
            error_capacity);
        if (outcome == G1FrameStageFiniteReject) {
#if defined(G1_FRAME_TRANSACTION_ENABLE_TEST_SEAM)
            if (trace_record != nullptr) {
                trace_record->common =
                    G1CandidateDispositionFiniteRejected;
            }
            g1_frame_candidate_trace_failure(
                trace_record, candidate_scratch);
#endif
            return G1CandidateFiniteRejected;
        }
        if (outcome != G1FrameStageContinue) {
#if defined(G1_FRAME_TRANSACTION_ENABLE_TEST_SEAM)
            if (trace_record != nullptr) {
                trace_record->common =
                    G1CandidateDispositionGlobalError;
            }
#endif
            return G1CandidateGlobalError;
        }
    }
    if (!g1_controller_state_is_valid(workspace.common_state)) {
#if defined(G1_FRAME_TRANSACTION_ENABLE_TEST_SEAM)
        if (trace_record != nullptr) {
            trace_record->common = G1CandidateDispositionGlobalError;
        }
#endif
        return G1CandidateGlobalError;
    }
#if defined(G1_FRAME_TRANSACTION_ENABLE_TEST_SEAM)
    if (trace_record != nullptr) {
        trace_record->common = G1CandidateDispositionAccepted;
    }
#endif

    if (!g1_controller_state_copy(
            workspace.raw_state,
            workspace.common_state,
            error,
            error_capacity)) {
#if defined(G1_FRAME_TRANSACTION_ENABLE_TEST_SEAM)
        if (trace_record != nullptr) {
            trace_record->raw = G1CandidateDispositionGlobalError;
        }
#endif
        return G1CandidateGlobalError;
    }
    candidate_scratch.rejection_branch = G1FrameCertificateRaw;
#if defined(G1_FRAME_TRANSACTION_ENABLE_TEST_SEAM)
    if (certification_trace != nullptr) {
        if (certification_trace->raw_evaluations == UINT32_MAX) {
            return G1CandidateGlobalError;
        }
        ++certification_trace->raw_evaluations;
    }
#endif
    for (int stage_index = G1FrameStageRawBegin;
         stage_index <= G1FrameStageRawPoseCertificate;
         ++stage_index) {
        const G1FrameStageOutcome outcome = g1_frame_run_one_stage(
            static_cast<G1FrameTransactionStage>(stage_index),
            run_stage,
            workspace.raw_state,
            candidate_scratch,
            external,
#if defined(G1_FRAME_TRANSACTION_ENABLE_TEST_SEAM)
            test_seam,
#endif
            error,
            error_capacity);
        if (outcome == G1FrameStageFiniteReject) {
#if defined(G1_FRAME_TRANSACTION_ENABLE_TEST_SEAM)
            if (trace_record != nullptr) {
                trace_record->raw =
                    G1CandidateDispositionFiniteRejected;
            }
            g1_frame_candidate_trace_failure(
                trace_record, candidate_scratch);
#endif
            return G1CandidateFiniteRejected;
        }
        if (outcome != G1FrameStageContinue) {
#if defined(G1_FRAME_TRANSACTION_ENABLE_TEST_SEAM)
            if (trace_record != nullptr) {
                trace_record->raw = G1CandidateDispositionGlobalError;
            }
#endif
            return G1CandidateGlobalError;
        }
    }
    if (!g1_frame_branch_certificate_matches_state(
            workspace.raw_state,
            candidate_scratch.raw_certificate)) {
#if defined(G1_FRAME_TRANSACTION_ENABLE_TEST_SEAM)
        if (trace_record != nullptr) {
            trace_record->raw = G1CandidateDispositionGlobalError;
        }
#endif
        return G1CandidateGlobalError;
    }
#if defined(G1_FRAME_TRANSACTION_ENABLE_TEST_SEAM)
    if (trace_record != nullptr) {
        trace_record->raw = G1CandidateDispositionAccepted;
    }
#endif

    if (!g1_controller_state_copy(
            workspace.ik_state,
            workspace.common_state,
            error,
            error_capacity)) {
#if defined(G1_FRAME_TRANSACTION_ENABLE_TEST_SEAM)
        if (trace_record != nullptr) {
            trace_record->ik = G1CandidateDispositionGlobalError;
        }
#endif
        return G1CandidateGlobalError;
    }
    candidate_scratch.rejection_branch = G1FrameCertificateIk;
#if defined(G1_FRAME_TRANSACTION_ENABLE_TEST_SEAM)
    if (certification_trace != nullptr) {
        if (certification_trace->ik_evaluations == UINT32_MAX) {
            return G1CandidateGlobalError;
        }
        ++certification_trace->ik_evaluations;
    }
#endif
    for (int stage_index = G1FrameStageIkBegin;
         stage_index <= G1FrameStageIkPoseCertificate;
         ++stage_index) {
        const G1FrameStageOutcome outcome = g1_frame_run_one_stage(
            static_cast<G1FrameTransactionStage>(stage_index),
            run_stage,
            workspace.ik_state,
            candidate_scratch,
            external,
#if defined(G1_FRAME_TRANSACTION_ENABLE_TEST_SEAM)
            test_seam,
#endif
            error,
            error_capacity);
        if (outcome == G1FrameStageFiniteReject) {
#if defined(G1_FRAME_TRANSACTION_ENABLE_TEST_SEAM)
            if (trace_record != nullptr) {
                trace_record->ik =
                    G1CandidateDispositionFiniteRejected;
            }
            g1_frame_candidate_trace_failure(
                trace_record, candidate_scratch);
#endif
            return G1CandidateFiniteRejected;
        }
        if (outcome != G1FrameStageContinue) {
#if defined(G1_FRAME_TRANSACTION_ENABLE_TEST_SEAM)
            if (trace_record != nullptr) {
                trace_record->ik = G1CandidateDispositionGlobalError;
            }
#endif
            return G1CandidateGlobalError;
        }
    }
    if (!g1_frame_branch_certificate_matches_state(
            workspace.ik_state,
            candidate_scratch.ik_certificate)) {
#if defined(G1_FRAME_TRANSACTION_ENABLE_TEST_SEAM)
        if (trace_record != nullptr) {
            trace_record->ik = G1CandidateDispositionGlobalError;
        }
#endif
        return G1CandidateGlobalError;
    }
#if defined(G1_FRAME_TRANSACTION_ENABLE_TEST_SEAM)
    if (trace_record != nullptr) {
        trace_record->ik = G1CandidateDispositionAccepted;
        if (!g1_frame_candidate_dual_outcome_is_valid(
                trace_record->common,
                trace_record->raw,
                trace_record->ik)) {
            return G1CandidateGlobalError;
        }
    }
#endif
    return G1CandidateDualAccepted;
}

static inline int g1_frame_candidate_source_range(
    const database& db,
    int frame)
{
    for (int range = 0; range < db.nranges(); ++range) {
        if (frame >= db.range_starts(range) &&
            frame < db.range_stops(range)) {
            return range;
        }
    }
    return -1;
}

static inline bool g1_frame_candidate_record_is_valid(
    const G1CandidateRecord& record,
    const database& db)
{
    if (record.kind < G1CandidateLegacy ||
        record.kind > G1CandidateIncumbent ||
        record.selected_frame < 0 ||
        record.selected_frame >= db.nframes() ||
        record.source_range < 0 ||
        record.source_range >= db.nranges() ||
        record.selected_frame < db.range_starts(record.source_range) ||
        record.selected_frame >= db.range_stops(record.source_range) ||
        !terrain_float_is_finite(record.selected_cost) ||
        record.selected_cost < 0.0f) {
        return false;
    }
    const int range_stop = db.range_stops(record.source_range);
    const int expected_executed =
        record.selected_frame < range_stop - 1
            ? record.selected_frame + 1
            : record.selected_frame;
    if (record.executed_frame != expected_executed) return false;
    if (record.kind == G1CandidateRecoveryTransition) {
        return record.transitioned &&
               record.recovery_rank < G1RecoveryTransitionCapacity;
    }
    if (record.kind == G1CandidateIncumbent) {
        return !record.transitioned &&
               record.recovery_rank == UINT32_MAX;
    }
    return record.recovery_rank == UINT32_MAX;
}

static inline bool g1_frame_recovery_request_matches_prefix(
    const G1RecoveryRequest& request,
    const G1FrameTransactionScratch& prefix_scratch,
    const g1_controller_state& immutable_baseline,
    const G1CandidateRecord& legacy_record,
    const G1FrameExternalInputs& external)
{
    if (!prefix_scratch.matching_scheduled ||
        !prefix_scratch.legacy_search_performed ||
        !prefix_scratch.recovery_request_ready ||
        request.db == nullptr ||
        request.db != external.db ||
        request.incumbent_frame != immutable_baseline.frame_index ||
        request.legacy_selected_frame != legacy_record.selected_frame ||
        !g1_frame_float_bits_equal(
            request.transition_cost, prefix_scratch.transition_cost) ||
        !g1_frame_float_bits_equal(
            request.public_incumbent_cost,
            immutable_baseline.incumbent_cost) ||
        !terrain_float_is_finite(request.transition_cost) ||
        request.transition_cost < 0.0f ||
        !terrain_float_is_finite(request.public_incumbent_cost) ||
        request.public_incumbent_cost < 0.0f ||
        request.ignore_range_end < 0 ||
        request.ignore_surrounding < 0 ||
        g1_frame_candidate_source_range(
            *request.db, request.incumbent_frame) < 0 ||
        g1_frame_candidate_source_range(
            *request.db, request.legacy_selected_frame) < 0) {
        return false;
    }
    for (uint32_t feature = 0U;
         feature < G1RecoveryFeatureCount;
         ++feature) {
        if (!terrain_float_is_finite(request.raw_query[feature]) ||
            !g1_frame_float_bits_equal(
                request.raw_query[feature],
                prefix_scratch.query[feature])) {
            return false;
        }
    }
    return true;
}

static inline bool g1_frame_recovery_set_is_valid(
    const G1RecoveryCandidateSet& set,
    const G1RecoveryRequest& request)
{
    if (request.db == nullptr ||
        set.count > G1RecoveryTailCapacity ||
        set.count >= G1CandidateAttemptCapacity ||
        set.work.accelerated_traversals != 1U ||
        set.work.full_scores_materialized != set.work.rows_tested ||
        set.work.rows_tested >
            static_cast<uint32_t>(request.db->nframes()) ||
        set.work.large_bounds_tested >
            static_cast<uint32_t>(request.db->nframes()) ||
        set.work.small_bounds_tested >
            static_cast<uint32_t>(request.db->nframes())) {
        return false;
    }

    uint32_t transition_count = set.count;
    const bool incumbent_expected =
        request.legacy_selected_frame != request.incumbent_frame;
    if (set.count > 0U &&
        set.records[set.count - 1U].kind == G1CandidateIncumbent) {
        --transition_count;
    }
    if ((incumbent_expected && transition_count + 1U != set.count) ||
        (!incumbent_expected && transition_count != set.count) ||
        transition_count > G1RecoveryTransitionCapacity ||
        set.work.rows_tested < transition_count) {
        return false;
    }

    for (uint32_t index = 0U; index < transition_count; ++index) {
        const G1CandidateRecord& record = set.records[index];
        if (!g1_frame_candidate_record_is_valid(
                record, *request.db) ||
            record.kind != G1CandidateRecoveryTransition ||
            record.recovery_rank != index) {
            return false;
        }
        const int range_start =
            request.db->range_starts(record.source_range);
        const int range_stop =
            request.db->range_stops(record.source_range);
        const int range_length = range_stop - range_start;
        const int search_stop =
            request.ignore_range_end < range_length
                ? range_stop - request.ignore_range_end
                : range_start;
        const int64_t incumbent_separation =
            record.selected_frame >= request.incumbent_frame
                ? static_cast<int64_t>(record.selected_frame) -
                      request.incumbent_frame
                : static_cast<int64_t>(request.incumbent_frame) -
                      record.selected_frame;
        if (record.selected_frame == request.incumbent_frame ||
            record.selected_frame == request.legacy_selected_frame ||
            record.selected_frame >= search_stop ||
            incumbent_separation <
                static_cast<int64_t>(request.ignore_surrounding) ||
            (record.selected_cost == 0.0f &&
             terrain_float_bits(record.selected_cost) != 0U) ||
            (terrain_float_bits(request.public_incumbent_cost) !=
                 terrain_float_bits(FLT_MAX) &&
             !(record.selected_cost < request.public_incumbent_cost))) {
            return false;
        }
        for (uint32_t previous = 0U; previous < index; ++previous) {
            if (set.records[previous].selected_frame ==
                record.selected_frame) {
                return false;
            }
        }
        if (index > 0U) {
            const G1CandidateRecord& previous = set.records[index - 1U];
            if (record.selected_cost < previous.selected_cost ||
                (record.selected_cost == previous.selected_cost &&
                 record.selected_frame < previous.selected_frame)) {
                return false;
            }
        }
    }

    if (incumbent_expected) {
        const G1CandidateRecord& incumbent =
            set.records[transition_count];
        const int source_range = g1_frame_candidate_source_range(
            *request.db, request.incumbent_frame);
        if (!g1_frame_candidate_record_is_valid(
                incumbent, *request.db) ||
            incumbent.kind != G1CandidateIncumbent ||
            incumbent.selected_frame != request.incumbent_frame ||
            incumbent.source_range != source_range ||
            !g1_frame_float_bits_equal(
                incumbent.selected_cost,
                request.public_incumbent_cost)) {
            return false;
        }
    }
    return true;
}

#if defined(G1_FRAME_TRANSACTION_ENABLE_TEST_SEAM)
static inline bool g1_frame_trace_append_attempt(
    G1CandidateCertificationTrace* trace,
    const G1CandidateRecord& candidate,
    G1CandidateScoreOwner score_owner,
    G1CandidateAttemptTraceRecord*& trace_record)
{
    trace_record = nullptr;
    if (trace == nullptr) return true;
    if (score_owner == G1CandidateScoreUnassigned ||
        trace->attempt_count >= G1CandidateAttemptCapacity) {
        return false;
    }
    trace_record = &trace->attempts[trace->attempt_count];
    g1_frame_attempt_trace_reset(*trace_record);
    trace_record->candidate = candidate;
    trace_record->score_owner = score_owner;
    ++trace->attempt_count;
    return true;
}
#endif

static inline bool g1_frame_candidate_failure_is_authentic(
    const G1FrameTransactionScratch& scratch)
{
    return scratch.requested_intent_ready &&
           g1_frame_intent_is_valid(scratch.requested_intent) &&
           g1_frame_rejection_is_valid(scratch.rejection) &&
           scratch.rejection.rejected &&
           g1_frame_rejection_matches_scratch_transaction(
               scratch.rejection, scratch);
}

static inline G1FrameTransactionStatus g1_frame_transaction_run(
    G1FrameRuntime& runtime,
    G1FrameStageRunner run_stage,
    G1RecoveryProvider recovery_provider,
    const G1FrameExternalInputs& external,
#if defined(G1_FRAME_TRANSACTION_ENABLE_TEST_SEAM)
    const G1FrameTransactionTestSeam* test_seam,
#endif
    char* error,
    int error_capacity)
{
    if (run_stage == nullptr || recovery_provider == nullptr ||
        !g1_frame_transaction_preflight(
            runtime,
            external,
#if defined(G1_FRAME_TRANSACTION_ENABLE_TEST_SEAM)
            test_seam,
#endif
            error,
            error_capacity)) {
        return G1FrameTransactionGlobalError;
    }

#if defined(G1_FRAME_TRANSACTION_ENABLE_TEST_SEAM)
    G1CandidateCertificationTrace* certification_trace =
        test_seam != nullptr
            ? test_seam->certification_trace
            : nullptr;
    if (certification_trace != nullptr) {
        g1_frame_certification_trace_reset(*certification_trace);
    }
#endif

    if (!g1_controller_state_copy(
            runtime.working_state,
            runtime.accepted_state,
            error,
            error_capacity) ||
        !g1_controller_state_is_valid(runtime.working_state)) {
        return G1FrameTransactionGlobalError;
    }

    G1FrameTransactionScratch prefix_scratch;
    prefix_scratch.prior_safe_stop_latched =
        runtime.publication.ik_safe_stop_latched;
    G1FrameTransactionScratch first_failure_scratch;
    bool first_failure_available = false;
    const auto retain_first_failure =
        [&](const G1FrameTransactionScratch& scratch) -> bool {
            if (!g1_frame_candidate_failure_is_authentic(scratch)) {
                return false;
            }
            if (!first_failure_available) {
                first_failure_scratch = scratch;
                first_failure_available = true;
            }
            return true;
        };
    const auto publish_first_failure =
        [&]() -> G1FrameTransactionStatus {
            if (!first_failure_available ||
                !g1_frame_publish_finite_rejection(
                    runtime,
                    first_failure_scratch,
                    external,
                    true)) {
                return G1FrameTransactionGlobalError;
            }
            return G1FrameTransactionFiniteRejected;
        };
    for (int stage_index = G1FrameStageInputRouteCommand;
         stage_index <= G1FrameStageMatcherSearch;
         ++stage_index) {
        const G1FrameStageOutcome outcome = g1_frame_run_one_stage(
            static_cast<G1FrameTransactionStage>(stage_index),
            run_stage,
            runtime.working_state,
            prefix_scratch,
            external,
#if defined(G1_FRAME_TRANSACTION_ENABLE_TEST_SEAM)
            test_seam,
#endif
            error,
            error_capacity);
        if (outcome == G1FrameStageFiniteReject) {
            if (!retain_first_failure(prefix_scratch)) {
                return G1FrameTransactionGlobalError;
            }
            return publish_first_failure();
        }
        if (outcome != G1FrameStageContinue) {
            return G1FrameTransactionGlobalError;
        }
    }
    if (!g1_controller_state_is_valid(runtime.working_state)) {
        return G1FrameTransactionGlobalError;
    }

    const bool matching_enabled =
        external.tuning.mode != G1_TestSequential;
    const bool matching_scheduled =
        matching_enabled && prefix_scratch.matching_scheduled;
    if (prefix_scratch.matching_scheduled != matching_scheduled ||
        prefix_scratch.legacy_search_performed != matching_scheduled ||
        prefix_scratch.recovery_request_ready != matching_scheduled) {
        return G1FrameTransactionGlobalError;
    }

    G1CandidateRecord slot_zero = prefix_scratch.slot_zero_record;
#if defined(G1_FRAME_TRANSACTION_ENABLE_TEST_SEAM)
    G1CandidateScoreOwner slot_zero_score_owner =
        G1CandidateScoreUnassigned;
#endif
    if (matching_scheduled) {
        slot_zero.kind = G1CandidateLegacy;
        slot_zero.recovery_rank = UINT32_MAX;
#if defined(G1_FRAME_TRANSACTION_ENABLE_TEST_SEAM)
        slot_zero_score_owner = G1CandidateScoreLegacy;
        if (certification_trace != nullptr) {
            certification_trace->legacy_traversals = 1U;
        }
#endif
    } else {
        slot_zero.kind = G1CandidateIncumbent;
        slot_zero.recovery_rank = UINT32_MAX;
        slot_zero.transitioned = false;
#if defined(G1_FRAME_TRANSACTION_ENABLE_TEST_SEAM)
        slot_zero_score_owner = G1CandidateScoreIncumbent;
#endif
    }
    prefix_scratch.slot_zero_record = slot_zero;

    const int baseline_source_range = g1_frame_candidate_source_range(
        *external.db, runtime.working_state.frame_index);
    if (!g1_frame_candidate_record_is_valid(
            slot_zero, *external.db) ||
        (!matching_scheduled &&
         (slot_zero.selected_frame !=
              runtime.working_state.frame_index ||
          slot_zero.source_range != baseline_source_range ||
          !g1_frame_float_bits_equal(
              slot_zero.selected_cost,
              runtime.working_state.incumbent_cost)))) {
        return G1FrameTransactionGlobalError;
    }

    G1FrameTransactionScratch candidate_scratch;

    const auto evaluate_record =
        [&](const G1CandidateRecord& record
#if defined(G1_FRAME_TRANSACTION_ENABLE_TEST_SEAM)
            , G1CandidateScoreOwner score_owner
#endif
            ) -> G1CandidateEvaluationOutcome {
#if defined(G1_FRAME_TRANSACTION_ENABLE_TEST_SEAM)
            G1CandidateAttemptTraceRecord* trace_record = nullptr;
            if (!g1_frame_trace_append_attempt(
                    certification_trace,
                    record,
                    score_owner,
                    trace_record)) {
                return G1CandidateGlobalError;
            }
#endif
            const G1CandidateEvaluationOutcome outcome =
                g1_frame_candidate_evaluate(
                    runtime.candidates,
                    candidate_scratch,
                    runtime.working_state,
                    prefix_scratch,
                    record,
                    run_stage,
                    external,
#if defined(G1_FRAME_TRANSACTION_ENABLE_TEST_SEAM)
                    trace_record,
                    test_seam,
#endif
                    error,
                    error_capacity);
            if (outcome == G1CandidateFiniteRejected) {
                if (!retain_first_failure(candidate_scratch)) {
                    return G1CandidateGlobalError;
                }
            }
            return outcome;
        };

    const auto commit_selected =
        [&]() -> G1FrameTransactionStatus {
            g1_controller_state& visible =
                external.tuning.ik_enabled
                    ? runtime.candidates.ik_state
                    : runtime.candidates.raw_state;
            if (!g1_controller_state_copy(
                    runtime.working_state,
                    visible,
                    error,
                    error_capacity)) {
                return G1FrameTransactionGlobalError;
            }
            runtime.working_state.ik =
                runtime.candidates.ik_state.ik;

            const G1FrameStageOutcome finalize_outcome =
                g1_frame_run_one_stage(
                    G1FrameStageAcceptedFinalize,
                    run_stage,
                    runtime.working_state,
                    candidate_scratch,
                    external,
#if defined(G1_FRAME_TRANSACTION_ENABLE_TEST_SEAM)
                    test_seam,
#endif
                    error,
                    error_capacity);
            if (finalize_outcome == G1FrameStageFiniteReject) {
                if (!retain_first_failure(candidate_scratch)) {
                    return G1FrameTransactionGlobalError;
                }
                return publish_first_failure();
            }
            if (finalize_outcome != G1FrameStageContinue) {
                return G1FrameTransactionGlobalError;
            }

            G1FramePublication publication_candidate;
            G1FrameAcceptedDiagnostic accepted_diagnostic_candidate;
            if (!g1_frame_success_candidates_are_valid(
                    runtime.working_state,
                    runtime.candidates.raw_state,
                    runtime.candidates.ik_state,
                    candidate_scratch,
                    external,
                    publication_candidate,
                    accepted_diagnostic_candidate)) {
                return G1FrameTransactionGlobalError;
            }
            g1_controller_state_swap(
                runtime.accepted_state, runtime.working_state);
            runtime.accepted_diagnostic =
                accepted_diagnostic_candidate;
            runtime.publication = publication_candidate;
            return G1FrameTransactionAccepted;
        };

    const G1CandidateEvaluationOutcome slot_zero_outcome =
        evaluate_record(
            slot_zero
#if defined(G1_FRAME_TRANSACTION_ENABLE_TEST_SEAM)
            , slot_zero_score_owner
#endif
        );
    if (slot_zero_outcome == G1CandidateGlobalError) {
        return G1FrameTransactionGlobalError;
    }
    if (slot_zero_outcome == G1CandidateDualAccepted) {
        return commit_selected();
    }
    if (!first_failure_available) {
        return G1FrameTransactionGlobalError;
    }

    if (!matching_scheduled) {
        return publish_first_failure();
    }

    const G1RecoveryRequest& recovery_request =
        prefix_scratch.recovery_request;
    if (!g1_frame_recovery_request_matches_prefix(
            recovery_request,
            prefix_scratch,
            runtime.working_state,
            slot_zero,
            external)) {
        return G1FrameTransactionGlobalError;
    }

#if defined(G1_FRAME_TRANSACTION_ENABLE_TEST_SEAM)
    if (certification_trace != nullptr) {
        certification_trace->recovery_request_available = true;
        certification_trace->recovery_request = recovery_request;
        certification_trace->recovery_provider_calls = 1U;
    }
#endif
    G1RecoveryCandidateSet recovery_set;
    const G1RecoveryProviderStatus provider_status =
        recovery_provider(
            recovery_set,
            recovery_request,
            error,
            error_capacity);
    if (provider_status != G1RecoveryProviderOk) {
        return G1FrameTransactionGlobalError;
    }
#if defined(G1_FRAME_TRANSACTION_ENABLE_TEST_SEAM)
    if (certification_trace != nullptr) {
        certification_trace->recovery_set = recovery_set;
    }
#endif
    if (!g1_frame_recovery_set_is_valid(
            recovery_set, recovery_request)) {
        return G1FrameTransactionGlobalError;
    }

    for (uint32_t index = 0U;
         index < recovery_set.count;
         ++index) {
        const G1CandidateRecord& record =
            recovery_set.records[index];
#if defined(G1_FRAME_TRANSACTION_ENABLE_TEST_SEAM)
        const G1CandidateScoreOwner score_owner =
            record.kind == G1CandidateRecoveryTransition
                ? G1CandidateScoreStrictRecovery
                : G1CandidateScoreIncumbent;
#endif
        const G1CandidateEvaluationOutcome outcome =
            evaluate_record(
                record
#if defined(G1_FRAME_TRANSACTION_ENABLE_TEST_SEAM)
                , score_owner
#endif
            );
        if (outcome == G1CandidateGlobalError) {
            return G1FrameTransactionGlobalError;
        }
        if (outcome == G1CandidateDualAccepted) {
            return commit_selected();
        }
    }

    return publish_first_failure();
}
static inline bool g1_frame_runtime_reset(
    G1FrameRuntime& output,
    const database& db,
    const terrain_support_set& support,
    const scene_pack& scene,
    const G1FrameResetConfig& config,
    char* error,
    int error_capacity)
{
    if (!g1_frame_reset_output_preflight(
            output,
            db,
            support,
            scene,
            config,
            error,
            error_capacity)) {
        return false;
    }
    if (!terrain_float_is_finite(config.initial_search_time) ||
        config.initial_search_time < 0.0f ||
        config.initial_search_time > 10.0f ||
        !g1_dt_is_exact_25_hz(config.dt) ||
        !terrain_float_is_positive_normal(
            config.trajectory_sample_time) ||
        config.route_mode != (config.route_id != nullptr) ||
        !g1_frame_artifacts_are_valid(
            db, support, scene, error, error_capacity)) {
        return scene_error(
            error,
            error_capacity,
            "frame runtime reset: invalid immutable configuration");
    }

    int route_index = -1;
    if (config.route_mode) {
        route_index = g1_frame_route_id_index(scene, config.route_id);
        if (route_index < 0) {
            return scene_error(
                error,
                error_capacity,
                "frame runtime reset: route ID is not a scene member");
        }
        const scene_route& route = scene.metadata.routes[
            static_cast<std::size_t>(route_index)];
        if (route.id.empty() || route.waypoints_xz.size() < 2U ||
            route.waypoints_xz.size() >
                static_cast<std::size_t>(INT_MAX)) {
            return scene_error(
                error,
                error_capacity,
                "frame runtime reset: route shape is invalid");
        }
        for (const std::pair<float, float>& waypoint :
             route.waypoints_xz) {
            if (!terrain_float_is_normal_or_zero_query(
                    waypoint.first) ||
                !terrain_float_is_normal_or_zero_query(
                    waypoint.second)) {
                return scene_error(
                    error,
                    error_capacity,
                    "frame runtime reset: route waypoint is invalid");
            }
        }
    }

    G1FrameRuntime candidate;
    if (!g1_controller_state_reset_configured(
            candidate.accepted_state,
            db,
            support,
            scene,
            config.initial_search_time,
            config.ik_enabled,
            config.dt,
            config.trajectory_sample_time,
            error,
            error_capacity) ||
        !g1_controller_state_reset_configured(
            candidate.working_state,
            db,
            support,
            scene,
            config.initial_search_time,
            config.ik_enabled,
            config.dt,
            config.trajectory_sample_time,
            error,
            error_capacity) ||
        !g1_controller_state_reset_configured(
            candidate.candidates.common_state,
            db,
            support,
            scene,
            config.initial_search_time,
            config.ik_enabled,
            config.dt,
            config.trajectory_sample_time,
            error,
            error_capacity) ||
        !g1_controller_state_reset_configured(
            candidate.candidates.raw_state,
            db,
            support,
            scene,
            config.initial_search_time,
            config.ik_enabled,
            config.dt,
            config.trajectory_sample_time,
            error,
            error_capacity) ||
        !g1_controller_state_reset_configured(
            candidate.candidates.ik_state,
            db,
            support,
            scene,
            config.initial_search_time,
            config.ik_enabled,
            config.dt,
            config.trajectory_sample_time,
            error,
            error_capacity)) {
        return false;
    }

    g1_controller_state* candidate_states[5] = {
        &candidate.accepted_state,
        &candidate.working_state,
        &candidate.candidates.common_state,
        &candidate.candidates.raw_state,
        &candidate.candidates.ik_state,
    };
    if (config.route_mode) {
        for (int state_index = 0; state_index < 5; ++state_index) {
            candidate_states[state_index]->route_index = route_index;
            candidate_states[state_index]->route_waypoint = 1;
            candidate_states[state_index]->route_frames = 0;
        }
    } else {
        for (int state_index = 0; state_index < 5; ++state_index) {
            candidate_states[state_index]->route_index = -1;
            candidate_states[state_index]->route_waypoint = 0;
            candidate_states[state_index]->route_frames = 0;
        }
    }

    for (int state_index = 0; state_index < 5; ++state_index) {
        if (!g1_frame_reset_candidate_is_valid(
                *candidate_states[state_index],
                config.initial_search_time)) {
            return scene_error(
                error,
                error_capacity,
                "frame runtime reset: independent state is invalid");
        }
    }
    if (!g1_frame_runtime_states_are_logically_equal(candidate)) {
        return scene_error(
            error,
            error_capacity,
            "frame runtime reset: independent states differ");
    }

    candidate.publication = G1FramePublication{};
    candidate.publication.requested_intent =
        candidate.accepted_state.command.intent;
    candidate.accepted_diagnostic = G1FrameAcceptedDiagnostic{};

    g1_controller_state_swap(
        output.accepted_state, candidate.accepted_state);
    g1_controller_state_swap(
        output.working_state, candidate.working_state);
    g1_controller_state_swap(
        output.candidates.common_state,
        candidate.candidates.common_state);
    g1_controller_state_swap(
        output.candidates.raw_state,
        candidate.candidates.raw_state);
    g1_controller_state_swap(
        output.candidates.ik_state,
        candidate.candidates.ik_state);
    output.publication = candidate.publication;
    output.accepted_diagnostic = candidate.accepted_diagnostic;
    return true;
}
