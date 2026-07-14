#pragma once

#if defined(G1_CLEARANCE_IMPLEMENTATION_TU)
#include "terrain_runtime.h"
struct G1LegConfig;
#else
#include "g1_ik.h"
#endif

#include <cstdint>

enum G1ClearanceStatus : uint8_t
{
    G1ClearanceOk = 0,
    G1ClearanceOutsideDomain,
    G1ClearanceBudgetExceeded,
    G1ClearanceUncertified,
    G1ClearanceInvalidInput,
    G1ClearanceInvalidField,
    G1ClearanceArithmeticFailure
};

struct G1ClearanceWitness
{
    double body_x = 0.0;
    double body_y = 0.0;
    double body_z = 0.0;
    double surface_x = 0.0;
    double surface_y = 0.0;
    double surface_z = 0.0;
    double segment_parameter = 0.0;
    double terrain_weight_0 = 0.0;
    double terrain_weight_1 = 0.0;
    double terrain_weight_2 = 0.0;
    uint32_t primitive_index = 0;
    int32_t cell_x = 0;
    int32_t cell_z = 0;
    uint32_t terrain_triangle_index = 0;
    uint32_t patch_index = 0;
    uint32_t candidate_kind = 0;
    uint32_t candidate_subindex = 0;
};

struct G1ClearanceWork
{
    uint32_t point_queries = 0;
    uint32_t cells_visited = 0;
    uint32_t primitive_triangle_pairs = 0;
    uint32_t face_patches = 0;
    uint32_t candidate_tests = 0;
    uint32_t subdivision_nodes = 0;
};

struct G1ClearanceResult
{
    double lower_bound_m = 0.0;
    double witness_upper_m = 0.0;
    G1ClearanceWitness witness;
    G1ClearanceWork work;
};

struct G1ClearanceBudget
{
    uint32_t maximum_point_queries;
    uint32_t maximum_cells;
    uint32_t maximum_primitive_triangle_pairs;
    uint32_t maximum_face_patches;
    uint32_t maximum_candidate_tests;
    uint32_t maximum_subdivision_nodes;
};

constexpr uint32_t G1ClearanceMaximumPointQueriesPerPose = 16;
constexpr uint32_t G1ClearanceMaximumCellsPerPrimitive = 512;
constexpr uint32_t G1ClearanceMaximumCellsPerPose = 2048;
constexpr uint32_t G1ClearanceMaximumCellsPerSwingFoot = 256;
constexpr uint32_t G1ClearanceMaximumPairsPerPrimitive = 1024;
constexpr uint32_t G1ClearanceMaximumPairsPerPose = 4096;
constexpr uint32_t G1ClearanceMaximumPairsPerSwingFoot = 512;
constexpr uint32_t G1ClearancePatchesPerPair = 8;
constexpr uint32_t G1ClearanceCandidatesPerPair = 32;
constexpr uint32_t G1ClearanceMaximumSubdivisionNodes = 8192;
constexpr double G1ClearanceMaximumCertificateWidthM = 1.0e-6;

struct G1SwingClearanceValidation
{
    double lower_margin_m = 0.0;
    double witness_upper_m = 0.0;
    bool sweep_evaluated = false;
    G1ClearanceWork work;
};

struct G1LegClearance
{
    G1ClearanceResult knee;
    G1ClearanceResult ankle;
    G1ClearanceResult toe;
    G1ClearanceResult foot;
    G1ClearanceResult thigh;
    G1ClearanceResult shin;
    G1ClearanceResult minimum;
};

struct G1PoseClearance
{
    G1ClearanceResult hips;
    G1LegClearance left;
    G1LegClearance right;
    G1ClearanceResult minimum;
};

struct G1SwingHistory
{
    bool initialized = false;
    vec3 previous_sphere_centers[4];
};

const char* g1_clearance_status_name(G1ClearanceStatus status);
bool g1_clearance_arithmetic_environment_is_supported();

G1ClearanceBudget g1_pose_clearance_budget();
G1ClearanceBudget g1_swing_foot_clearance_budget();

G1ClearanceStatus g1_apply_swing_lift_y(
    float& output_y,
    float input_y,
    float lift_m,
    char* error,
    int error_capacity);

G1ClearanceStatus g1_point_clearance(
    G1ClearanceResult& output,
    const G1ClearanceBudget& limits,
    const heightfield& field,
    vec3 point,
    char* error,
    int error_capacity);

G1ClearanceStatus g1_sphere_clearance(
    G1ClearanceResult& output,
    const G1ClearanceBudget& limits,
    const heightfield& field,
    vec3 center,
    float radius_m,
    char* error,
    int error_capacity);

G1ClearanceStatus g1_capsule_clearance(
    G1ClearanceResult& output,
    const G1ClearanceBudget& limits,
    const heightfield& field,
    vec3 endpoint_a,
    vec3 endpoint_b,
    float radius_m,
    char* error,
    int error_capacity);

G1ClearanceStatus g1_foot_clearance(
    G1ClearanceResult& output,
    const G1ClearanceBudget& limits,
    const heightfield& field,
    const vec3 sphere_centers[4],
    float radius_m,
    char* error,
    int error_capacity);

G1ClearanceStatus g1_swept_foot_clearance(
    G1ClearanceResult& output,
    const G1ClearanceBudget& limits,
    const heightfield& field,
    const vec3 previous_centers[4],
    const vec3 current_centers[4],
    float radius_m,
    char* error,
    int error_capacity);

G1ClearanceStatus g1_measure_leg_clearance(
    G1LegClearance& output,
    const G1ClearanceBudget& limits,
    const heightfield& field,
    const slice1d<vec3> global_positions,
    const slice1d<quat> global_rotations,
    const G1LegConfig& config,
    char* error,
    int error_capacity);

G1ClearanceStatus g1_measure_pose_clearance(
    G1PoseClearance& output,
    const G1ClearanceBudget& limits,
    const heightfield& field,
    const slice1d<vec3> global_positions,
    const slice1d<quat> global_rotations,
    char* error,
    int error_capacity);

bool g1_swing_history_reset(
    G1SwingHistory& output,
    const vec3 sphere_centers[4],
    char* error,
    int error_capacity);

bool g1_swing_history_commit(
    G1SwingHistory& history,
    const vec3 accepted_sphere_centers[4],
    char* error,
    int error_capacity);

G1ClearanceStatus g1_swing_clearance_validate(
    G1SwingClearanceValidation& output,
    const G1ClearanceBudget& limits,
    const G1SwingHistory& history,
    const heightfield& field,
    const G1LegConfig& config,
    const vec3 final_sphere_centers[4],
    bool recorded_contact,
    float dt,
    char* error,
    int error_capacity);
