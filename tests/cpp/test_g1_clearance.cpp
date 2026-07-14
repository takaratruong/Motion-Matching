#include "g1_clearance.h"

#include <cfenv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <type_traits>
#include <utility>

#if defined(__SSE__) || defined(_M_X64) || defined(_M_IX86_FP)
#include <xmmintrin.h>
#endif

static void check(bool condition, const char* message)
{
    if (!condition) {
        std::fprintf(stderr, "G1 clearance test failed: %s\n", message);
        std::exit(1);
    }
}

static_assert(
    std::is_same<
        std::underlying_type<G1ClearanceStatus>::type,
        uint8_t>::value,
    "clearance status has an exact byte representation");
static_assert(G1ClearanceOk == 0, "Ok status value");
static_assert(G1ClearanceOutsideDomain == 1, "outside status value");
static_assert(G1ClearanceBudgetExceeded == 2, "budget status value");
static_assert(G1ClearanceUncertified == 3, "uncertified status value");
static_assert(G1ClearanceInvalidInput == 4, "input status value");
static_assert(G1ClearanceInvalidField == 5, "field status value");
static_assert(G1ClearanceArithmeticFailure == 6, "arithmetic status value");

static_assert(sizeof(float) == 4, "test requires binary32");
static_assert(sizeof(double) == 8, "test requires binary64");
static_assert(std::numeric_limits<float>::is_iec559,
              "test requires IEEE binary32");
static_assert(std::numeric_limits<double>::is_iec559,
              "test requires IEEE binary64");
static_assert(
    std::numeric_limits<float>::has_denorm == std::denorm_present,
    "test requires binary32 gradual underflow");
static_assert(
    std::numeric_limits<double>::has_denorm == std::denorm_present,
    "test requires binary64 gradual underflow");

static_assert(
    std::is_same<
        decltype(std::declval<G1ClearanceResult>().lower_bound_m),
        double>::value,
    "clearance lower bound is binary64");
static_assert(
    std::is_same<
        decltype(std::declval<G1ClearanceResult>().witness_upper_m),
        double>::value,
    "clearance witness upper is binary64");
static_assert(
    std::is_same<
        decltype(std::declval<G1ClearanceWitness>().body_y),
        double>::value,
    "witness geometry is binary64");
static_assert(
    std::is_same<
        decltype(std::declval<G1ClearanceWitness>().terrain_weight_2),
        double>::value,
    "witness weights are binary64");
static_assert(
    std::is_same<
        decltype(std::declval<G1SwingClearanceValidation>().lower_margin_m),
        double>::value,
    "swing margin is binary64");
static_assert(
    std::is_same<
        decltype(std::declval<G1SwingClearanceValidation>().witness_upper_m),
        double>::value,
    "swing witness is binary64");

static_assert(G1ClearanceMaximumPointQueriesPerPose == 16,
              "point-query cap");
static_assert(G1ClearanceMaximumCellsPerPrimitive == 512,
              "primitive-cell cap");
static_assert(G1ClearanceMaximumCellsPerPose == 2048,
              "pose-cell cap");
static_assert(G1ClearanceMaximumCellsPerSwingFoot == 256,
              "swing-cell cap");
static_assert(G1ClearanceMaximumPairsPerPrimitive == 1024,
              "primitive-pair cap");
static_assert(G1ClearanceMaximumPairsPerPose == 4096,
              "pose-pair cap");
static_assert(G1ClearanceMaximumPairsPerSwingFoot == 512,
              "swing-pair cap");
static_assert(G1ClearancePatchesPerPair == 8,
              "patch count per pair");
static_assert(G1ClearanceCandidatesPerPair == 32,
              "candidate count per pair");
static_assert(G1ClearanceMaximumSubdivisionNodes == 8192,
              "subdivision cap");
static_assert(G1ClearanceMaximumCertificateWidthM == 1.0e-6,
              "certificate-width cap");
static_assert(
    G1ClearanceMaximumPairsPerPose * G1ClearancePatchesPerPair == 32768,
    "pose patch cap product");
static_assert(
    G1ClearanceMaximumPairsPerPose * G1ClearanceCandidatesPerPair ==
        131072,
    "pose candidate cap product");
static_assert(
    G1ClearanceMaximumPairsPerSwingFoot *
        G1ClearancePatchesPerPair == 4096,
    "swing patch cap product");
static_assert(
    G1ClearanceMaximumPairsPerSwingFoot *
        G1ClearanceCandidatesPerPair == 16384,
    "swing candidate cap product");

using G1ApplySwingLiftYSignature = G1ClearanceStatus (*)(
    float&, float, float, char*, int);
using G1StatusNameSignature = const char* (*)(G1ClearanceStatus);
using G1EnvironmentSignature = bool (*)();
using G1HistoryResetSignature = bool (*)(
    G1SwingHistory&, const vec3*, char*, int);
using G1HistoryCommitSignature = bool (*)(
    G1SwingHistory&, const vec3*, char*, int);
using G1SwingValidateSignature = G1ClearanceStatus (*)(
    G1SwingClearanceValidation&,
    const G1ClearanceBudget&,
    const G1SwingHistory&,
    const heightfield&,
    const G1LegConfig&,
    const vec3*,
    bool,
    float,
    char*,
    int);

static_assert(
    std::is_same<
        decltype(&g1_apply_swing_lift_y),
        G1ApplySwingLiftYSignature>::value,
    "sole-command materializer signature");
static_assert(
    std::is_same<
        decltype(&g1_clearance_status_name),
        G1StatusNameSignature>::value,
    "status-name signature");
static_assert(
    std::is_same<
        decltype(&g1_clearance_arithmetic_environment_is_supported),
        G1EnvironmentSignature>::value,
    "non-inline environment probe signature");
static_assert(
    std::is_same<
        decltype(&g1_swing_history_reset),
        G1HistoryResetSignature>::value,
    "history reset uses a non-copying center pointer");
static_assert(
    std::is_same<
        decltype(&g1_swing_history_commit),
        G1HistoryCommitSignature>::value,
    "history commit uses a non-copying center pointer");
static_assert(
    std::is_same<
        decltype(&g1_swing_clearance_validate),
        G1SwingValidateSignature>::value,
    "swing validation consumes const history by reference");

static float float_from_bits(uint32_t bits)
{
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

static uint32_t float_bits(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static uint64_t double_bits(double value)
{
    uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static bool clearance_work_same(
    const G1ClearanceWork& left,
    const G1ClearanceWork& right)
{
    return left.point_queries == right.point_queries &&
           left.cells_visited == right.cells_visited &&
           left.primitive_triangle_pairs ==
               right.primitive_triangle_pairs &&
           left.face_patches == right.face_patches &&
           left.candidate_tests == right.candidate_tests &&
           left.subdivision_nodes == right.subdivision_nodes;
}

static bool clearance_witness_same(
    const G1ClearanceWitness& left,
    const G1ClearanceWitness& right)
{
    return double_bits(left.body_x) == double_bits(right.body_x) &&
           double_bits(left.body_y) == double_bits(right.body_y) &&
           double_bits(left.body_z) == double_bits(right.body_z) &&
           double_bits(left.surface_x) == double_bits(right.surface_x) &&
           double_bits(left.surface_y) == double_bits(right.surface_y) &&
           double_bits(left.surface_z) == double_bits(right.surface_z) &&
           double_bits(left.segment_parameter) ==
               double_bits(right.segment_parameter) &&
           double_bits(left.terrain_weight_0) ==
               double_bits(right.terrain_weight_0) &&
           double_bits(left.terrain_weight_1) ==
               double_bits(right.terrain_weight_1) &&
           double_bits(left.terrain_weight_2) ==
               double_bits(right.terrain_weight_2) &&
           left.primitive_index == right.primitive_index &&
           left.cell_x == right.cell_x &&
           left.cell_z == right.cell_z &&
           left.terrain_triangle_index ==
               right.terrain_triangle_index &&
           left.patch_index == right.patch_index &&
           left.candidate_kind == right.candidate_kind &&
           left.candidate_subindex == right.candidate_subindex;
}

static bool clearance_result_same(
    const G1ClearanceResult& left,
    const G1ClearanceResult& right)
{
    return double_bits(left.lower_bound_m) ==
               double_bits(right.lower_bound_m) &&
           double_bits(left.witness_upper_m) ==
               double_bits(right.witness_upper_m) &&
           clearance_witness_same(left.witness, right.witness) &&
           clearance_work_same(left.work, right.work);
}

template<typename T>
struct ByteSnapshot
{
    unsigned char bytes[sizeof(T)];

    explicit ByteSnapshot(const T& value)
    {
        std::memcpy(bytes, &value, sizeof(value));
    }

    bool same(const T& value) const
    {
        return std::memcmp(bytes, &value, sizeof(value)) == 0;
    }
};

static void query_parity_make_field(heightfield& field)
{
    field.version = 2;
    field.nx = 2;
    field.nz = 2;
    field.origin_x = 0.0f;
    field.origin_z = 0.0f;
    field.cell_size = 1.0f;
    field.exterior_height = -10.0f;
    field.heights.resize(4);
    field.heights(0) = 0.0f;
    field.heights(1) = 2.0f;
    field.heights(2) = 4.0f;
    field.heights(3) = 10.0f;
}

static void query_parity_emit(
    const char* name,
    const heightfield& field,
    float x,
    float z,
    G1SurfaceQueryStatus expected_status,
    uint32_t expected_height_bits)
{
    G1SurfaceSample sample = {
        float_from_bits(UINT32_C(0x41234567)),
        vec3(
            float_from_bits(UINT32_C(0x3f123456)),
            float_from_bits(UINT32_C(0x3f234567)),
            float_from_bits(UINT32_C(0x3f345678)))
    };
    const ByteSnapshot<G1SurfaceSample> before(sample);
    const G1SurfaceQueryStatus status =
        g1_surface_query_v2(sample, field, x, z);
    check(status == expected_status,
          "query parity fixture has its locked semantic status");
    if (status == G1SurfaceQueryValid) {
        check(float_bits(sample.height) == expected_height_bits,
              "query parity fixture has its locked height bits");
    } else {
        check(before.same(sample),
              "failed query parity fixture preserves seeded output bytes");
    }
    std::printf(
        "query=%s status=%u height=%08x normal=%08x,%08x,%08x\n",
        name, static_cast<unsigned int>(status),
        float_bits(sample.height),
        float_bits(sample.normal.x),
        float_bits(sample.normal.y),
        float_bits(sample.normal.z));
}

static int run_query_parity_mode()
{
    heightfield field;
    query_parity_make_field(field);
    query_parity_emit(
        "t0", field, 0.75f, 0.25f,
        G1SurfaceQueryValid, UINT32_C(0x40600000));
    query_parity_emit(
        "t1", field, 0.25f, 0.75f,
        G1SurfaceQueryValid, UINT32_C(0x40900000));
    query_parity_emit(
        "tie", field, 0.50f, 0.50f,
        G1SurfaceQueryValid, UINT32_C(0x40a00000));
    query_parity_emit(
        "maximum-edge", field, 1.0f, 1.0f,
        G1SurfaceQueryValid, UINT32_C(0x41200000));
    query_parity_emit(
        "outside", field,
        float_from_bits(UINT32_C(0x3f800001)), 0.5f,
        G1SurfaceQueryOutside, 0);

    field.version = 1;
    query_parity_emit(
        "invalid-field", field, 0.5f, 0.5f,
        G1SurfaceQueryInvalid, 0);
    field.version = 2;
    field.heights(3) = float_from_bits(UINT32_C(0x7fc00001));
    query_parity_emit(
        "invalid-height", field, 0.75f, 0.25f,
        G1SurfaceQueryInvalid, 0);
    return 0;
}

static G1ClearanceResult seeded_result(double key)
{
    G1ClearanceResult output = {};
    output.lower_bound_m = key;
    output.witness_upper_m = key + 1.0;
    output.witness.body_x = key + 2.0;
    output.witness.surface_y = key + 3.0;
    output.witness.primitive_index = UINT32_C(0x10203040);
    output.witness.cell_x = -37;
    output.work.point_queries = UINT32_C(0x11223344);
    output.work.subdivision_nodes = UINT32_C(0x55667788);
    return output;
}

struct PublicOutputs
{
    float output_y;
    G1ClearanceResult result;
    G1LegClearance leg;
    G1PoseClearance pose;
    G1SwingClearanceValidation swing;
};

static PublicOutputs seeded_public_outputs()
{
    PublicOutputs outputs = {};
    outputs.output_y = float_from_bits(UINT32_C(0x41234567));
    outputs.result = seeded_result(11.0);
    outputs.leg.minimum = seeded_result(21.0);
    outputs.pose.minimum = seeded_result(31.0);
    outputs.swing.lower_margin_m = 41.0;
    outputs.swing.witness_upper_m = 42.0;
    outputs.swing.sweep_evaluated = true;
    outputs.swing.work.candidate_tests = UINT32_C(0x12345678);
    return outputs;
}

enum PublicStatusEntry
{
    PublicLift = 0,
    PublicPoint,
    PublicSphere,
    PublicCapsule,
    PublicFoot,
    PublicSweptFoot,
    PublicMeasureLeg,
    PublicMeasurePose,
    PublicSwingValidate,
    PublicStatusEntryCount
};

static G1ClearanceStatus invoke_public_status_entry(
    PublicStatusEntry entry,
    PublicOutputs& outputs,
    const G1ClearanceBudget& limits,
    char* error,
    int error_capacity)
{
    heightfield field;
    const G1LegConfig config = g1_left_leg_config();
    G1SwingHistory history = {};
    const float hostile = float_from_bits(UINT32_C(0x7fc00001));
    switch (entry) {
    case PublicLift:
        return g1_apply_swing_lift_y(
            outputs.output_y, hostile, hostile,
            error, error_capacity);
    case PublicPoint:
        return g1_point_clearance(
            outputs.result, limits, field,
            vec3(hostile, hostile, hostile), error, error_capacity);
    case PublicSphere:
        return g1_sphere_clearance(
            outputs.result, limits, field,
            vec3(hostile, hostile, hostile), hostile,
            error, error_capacity);
    case PublicCapsule:
        return g1_capsule_clearance(
            outputs.result, limits, field,
            vec3(hostile, hostile, hostile),
            vec3(hostile, hostile, hostile), hostile,
            error, error_capacity);
    case PublicFoot:
        return g1_foot_clearance(
            outputs.result, limits, field, NULL, hostile,
            error, error_capacity);
    case PublicSweptFoot:
        return g1_swept_foot_clearance(
            outputs.result, limits, field, NULL, NULL, hostile,
            error, error_capacity);
    case PublicMeasureLeg:
        return g1_measure_leg_clearance(
            outputs.leg, limits, field,
            slice1d<vec3>(0, NULL), slice1d<quat>(0, NULL),
            config, error, error_capacity);
    case PublicMeasurePose:
        return g1_measure_pose_clearance(
            outputs.pose, limits, field,
            slice1d<vec3>(0, NULL), slice1d<quat>(0, NULL),
            error, error_capacity);
    case PublicSwingValidate:
        return g1_swing_clearance_validate(
            outputs.swing, limits, history, field, config, NULL,
            false, hostile, error, error_capacity);
    default:
        return G1ClearanceInvalidInput;
    }
}

struct PublicProtectedSpan
{
    char* data;
    int size;
};

static PublicProtectedSpan public_output_span(
    PublicStatusEntry entry,
    PublicOutputs& outputs)
{
    switch (entry) {
    case PublicLift:
        return {
            reinterpret_cast<char*>(&outputs.output_y),
            static_cast<int>(sizeof(outputs.output_y))
        };
    case PublicPoint:
    case PublicSphere:
    case PublicCapsule:
    case PublicFoot:
    case PublicSweptFoot:
        return {
            reinterpret_cast<char*>(&outputs.result),
            static_cast<int>(sizeof(outputs.result))
        };
    case PublicMeasureLeg:
        return {
            reinterpret_cast<char*>(&outputs.leg),
            static_cast<int>(sizeof(outputs.leg))
        };
    case PublicMeasurePose:
        return {
            reinterpret_cast<char*>(&outputs.pose),
            static_cast<int>(sizeof(outputs.pose))
        };
    case PublicSwingValidate:
        return {
            reinterpret_cast<char*>(&outputs.swing),
            static_cast<int>(sizeof(outputs.swing))
        };
    default:
        return {NULL, 0};
    }
}

static bool public_entry_uses_swing_budget(PublicStatusEntry entry)
{
    return entry == PublicSweptFoot || entry == PublicSwingValidate;
}

static G1ClearanceBudget public_entry_budget(PublicStatusEntry entry)
{
    return public_entry_uses_swing_budget(entry)
        ? g1_swing_foot_clearance_budget()
        : g1_pose_clearance_budget();
}

static void test_status_and_factory_contract()
{
    const char* const expected_names[] = {
        "ok",
        "outside-domain",
        "budget-exceeded",
        "uncertified",
        "invalid-input",
        "invalid-field",
        "arithmetic-failure"
    };
    for (int status = G1ClearanceOk;
         status <= G1ClearanceArithmeticFailure;
         ++status) {
        check(std::strcmp(
                  g1_clearance_status_name(
                      static_cast<G1ClearanceStatus>(status)),
                  expected_names[status]) == 0,
              "status name matches its locked enum value");
    }
    check(std::strcmp(
              g1_clearance_status_name(
                  static_cast<G1ClearanceStatus>(UINT8_C(255))),
              "unknown") == 0,
          "unknown status has a stable name");

    const G1ClearanceBudget pose = g1_pose_clearance_budget();
    check(pose.maximum_point_queries == 16 &&
          pose.maximum_cells == 2048 &&
          pose.maximum_primitive_triangle_pairs == 4096 &&
          pose.maximum_face_patches == 32768 &&
          pose.maximum_candidate_tests == 131072 &&
          pose.maximum_subdivision_nodes == 8192,
          "pose factory has exact immutable ceilings");
    const G1ClearanceBudget swing = g1_swing_foot_clearance_budget();
    check(swing.maximum_point_queries == 0 &&
          swing.maximum_cells == 256 &&
          swing.maximum_primitive_triangle_pairs == 512 &&
          swing.maximum_face_patches == 4096 &&
          swing.maximum_candidate_tests == 16384 &&
          swing.maximum_subdivision_nodes == 8192,
          "swing factory has exact immutable ceilings");
}

using BudgetMember = uint32_t G1ClearanceBudget::*;

static G1ClearanceStatus invoke_budget_stub(
    bool swing_family,
    G1ClearanceResult& output,
    const G1ClearanceBudget& limits,
    char* error,
    int error_capacity)
{
    heightfield field;
    if (swing_family) {
        return g1_swept_foot_clearance(
            output, limits, field, NULL, NULL, 0.02f,
            error, error_capacity);
    }
    return g1_sphere_clearance(
        output, limits, field, vec3(), 0.02f,
        error, error_capacity);
}

static void require_budget_status(
    bool swing_family,
    const G1ClearanceBudget& limits,
    G1ClearanceStatus expected,
    const char* message)
{
    G1ClearanceResult output = seeded_result(51.0);
    const ByteSnapshot<G1ClearanceResult> output_before(output);
    const ByteSnapshot<G1ClearanceBudget> limits_before(limits);
    char error[128] = {};
    const G1ClearanceStatus status = invoke_budget_stub(
        swing_family, output, limits,
        error, static_cast<int>(sizeof(error)));
    check(status == expected, message);
    check(output_before.same(output),
          "budget status leaves seeded result unchanged");
    check(limits_before.same(limits),
          "budget status never mutates caller limits");
}

static void test_budget_caps_for_family(bool swing_family)
{
    const G1ClearanceBudget factory = swing_family
        ? g1_swing_foot_clearance_budget()
        : g1_pose_clearance_budget();
    const BudgetMember fields[] = {
        &G1ClearanceBudget::maximum_point_queries,
        &G1ClearanceBudget::maximum_cells,
        &G1ClearanceBudget::maximum_primitive_triangle_pairs,
        &G1ClearanceBudget::maximum_face_patches,
        &G1ClearanceBudget::maximum_candidate_tests,
        &G1ClearanceBudget::maximum_subdivision_nodes
    };
    for (size_t index = 0;
         index < sizeof(fields) / sizeof(fields[0]);
         ++index) {
        G1ClearanceBudget limits = factory;
        require_budget_status(
            swing_family, limits, G1ClearanceUncertified,
            "factory cap is admitted by its family");

        limits = factory;
        limits.*fields[index] = 0;
        require_budget_status(
            swing_family, limits, G1ClearanceUncertified,
            "zero is an admitted tightened budget");

        limits = factory;
        limits.*fields[index] = factory.*fields[index] + 1;
        require_budget_status(
            swing_family, limits, G1ClearanceInvalidInput,
            "factory one-over is rejected without clamping");

        limits = factory;
        limits.*fields[index] = UINT32_MAX;
        require_budget_status(
            swing_family, limits, G1ClearanceInvalidInput,
            "UINT32_MAX budget is rejected without wrapping");
    }
}

static void test_budget_contract()
{
    test_budget_caps_for_family(false);
    test_budget_caps_for_family(true);
}

static void test_swing_lift_materializer()
{
    char error[128] = {};
    float output = float_from_bits(UINT32_C(0x41234567));

    const float input = float_from_bits(UINT32_C(0x3d0f5c28));
    check(g1_apply_swing_lift_y(
              output, input, float_from_bits(UINT32_C(0x31000001)),
              error, static_cast<int>(sizeof(error))) == G1ClearanceOk,
          error);
    check(float_bits(output) == UINT32_C(0x3d0f5c29),
          "strict one-round lift materializes the upper tie fixture");

    output = float_from_bits(UINT32_C(0x41234567));
    check(g1_apply_swing_lift_y(
              output, input, float_from_bits(UINT32_C(0x31000000)),
              error, static_cast<int>(sizeof(error))) == G1ClearanceOk,
          error);
    check(float_bits(output) == UINT32_C(0x3d0f5c28),
          "strict one-round lift materializes the lower tie fixture");

    output = float_from_bits(UINT32_C(0x41234567));
    check(g1_apply_swing_lift_y(
              output, -0.0f, -0.0f,
              error, static_cast<int>(sizeof(error))) == G1ClearanceOk,
          error);
    check(float_bits(output) == 0,
          "signed-zero input and lift canonicalize to positive zero");

    const float denormal = std::numeric_limits<float>::denorm_min();
    output = float_from_bits(UINT32_C(0x41234567));
    const uint32_t output_before = float_bits(output);
    check(g1_apply_swing_lift_y(
              output, 0.0f, denormal,
              error, static_cast<int>(sizeof(error))) ==
              G1ClearanceArithmeticFailure,
          "nonzero-subnormal materialized command fails closed");
    check(float_bits(output) == output_before,
          "subnormal materialization leaves scalar output unchanged");

    const float invalid_lifts[] = {
        -0.001f,
        float_from_bits(UINT32_C(0x3da3d70b)),
        std::numeric_limits<float>::infinity(),
        float_from_bits(UINT32_C(0x7fc00001))
    };
    for (size_t index = 0;
         index < sizeof(invalid_lifts) / sizeof(invalid_lifts[0]);
         ++index) {
        output = float_from_bits(UINT32_C(0x41234567));
        const uint32_t before = float_bits(output);
        check(g1_apply_swing_lift_y(
                  output, input, invalid_lifts[index],
                  error, static_cast<int>(sizeof(error))) ==
                  G1ClearanceInvalidInput,
              "invalid lift is rejected");
        check(float_bits(output) == before,
              "invalid lift leaves scalar output unchanged");
    }

    output = float_from_bits(UINT32_C(0x41234567));
    check(g1_apply_swing_lift_y(
              output, denormal, 0.0f, NULL, 0) ==
              G1ClearanceInvalidInput,
          "subnormal input is rejected with a null diagnostic");
    check(float_bits(output) == UINT32_C(0x41234567),
          "null diagnostic failure is transactional");

    char one_byte[1] = {'x'};
    check(g1_apply_swing_lift_y(
              output, input, -1.0f, one_byte, 1) ==
              G1ClearanceInvalidInput,
          "one-byte diagnostic remains safe");
    check(one_byte[0] == '\0',
          "one-byte diagnostic is terminated");

    char untouched = 'q';
    check(g1_apply_swing_lift_y(
              output, input, -1.0f, &untouched, 0) ==
              G1ClearanceInvalidInput,
          "zero-capacity diagnostic remains safe");
    check(untouched == 'q',
          "zero-capacity diagnostic is untouched");
}

static void point_make_field(
    heightfield& field,
    int nx = 2,
    int nz = 2,
    float origin_x = 0.0f,
    float origin_z = 0.0f,
    float cell_size = 1.0f,
    float exterior_height = -10.0f)
{
    field.version = 2;
    field.nx = nx;
    field.nz = nz;
    field.origin_x = origin_x;
    field.origin_z = origin_z;
    field.cell_size = cell_size;
    field.exterior_height = exterior_height;
    field.heights.resize(nx * nz);
    field.heights.set(0.0f);
}

static G1ClearanceResult require_point_ok(
    const heightfield& field,
    vec3 point,
    const G1ClearanceBudget& limits)
{
    G1ClearanceResult output = seeded_result(71.0);
    char error[256] = {};
    const G1ClearanceStatus status = g1_point_clearance(
        output, limits, field, point,
        error, static_cast<int>(sizeof(error)));
    check(status == G1ClearanceOk,
          error[0] == '\0' ? "point clearance must certify" : error);
    check(output.lower_bound_m <= output.witness_upper_m &&
          output.witness_upper_m - output.lower_bound_m <=
              G1ClearanceMaximumCertificateWidthM,
          "point result has a binary64 certificate width");

    G1SurfaceSample producer = {};
    check(g1_surface_query_v2(
              producer, field, point.x, point.z) ==
              G1SurfaceQueryValid,
          "point fixture is accepted by the semantic producer");
    const volatile double producer_clearance =
        static_cast<double>(point.y) -
        static_cast<double>(producer.height);
    check(output.lower_bound_m <= producer_clearance,
          "point lower bound never exceeds producer clearance");
    return output;
}

static G1ClearanceResult require_point_ok(
    const heightfield& field,
    vec3 point)
{
    const G1ClearanceBudget limits = g1_pose_clearance_budget();
    return require_point_ok(field, point, limits);
}

static void require_point_failure(
    const heightfield& field,
    vec3 point,
    const G1ClearanceBudget& limits,
    G1ClearanceStatus expected,
    const char* message)
{
    G1ClearanceResult output = seeded_result(73.0);
    const ByteSnapshot<G1ClearanceResult> output_before(output);
    const ByteSnapshot<G1ClearanceBudget> limits_before(limits);
    char error[128] = {};
    const G1ClearanceStatus status = g1_point_clearance(
        output, limits, field, point,
        error, static_cast<int>(sizeof(error)));
    check(status == expected, message);
    check(output_before.same(output),
          "failed point query preserves every output byte");
    check(limits_before.same(limits),
          "failed point query preserves immutable limits");
}

static void require_point_failure(
    const heightfield& field,
    vec3 point,
    G1ClearanceStatus expected,
    const char* message)
{
    const G1ClearanceBudget limits = g1_pose_clearance_budget();
    require_point_failure(field, point, limits, expected, message);
}

static void test_point_fixed_diagonal_and_determinism()
{
    heightfield field;
    point_make_field(field);
    field.heights(0) = 0.0f;
    field.heights(1) = 2.0f;
    field.heights(2) = 4.0f;
    field.heights(3) = 10.0f;

    const vec3 points[] = {
        vec3(0.75f, 20.0f, 0.25f),
        vec3(0.25f, 20.0f, 0.75f),
        vec3(0.50f, 20.0f, 0.50f)
    };
    const double expected_heights[] = {3.5, 4.5, 5.0};
    const uint32_t expected_triangles[] = {0, 1, 0};
    const double expected_weights[][3] = {
        {0.25, 0.50, 0.25},
        {0.25, 0.25, 0.50},
        {0.50, 0.00, 0.50}
    };
    const double mandatory_ulp =
        static_cast<double>(
            float_from_bits(float_bits(10.0f) + 1)) - 10.0;

    for (size_t index = 0;
         index < sizeof(points) / sizeof(points[0]);
         ++index) {
        const G1ClearanceResult first =
            require_point_ok(field, points[index]);
        const G1ClearanceResult second =
            require_point_ok(field, points[index]);
        check(clearance_result_same(first, second),
              "point result, witness, and work are deterministic");

        const double expected_clearance =
            static_cast<double>(points[index].y) -
            expected_heights[index];
        check(first.witness_upper_m >= expected_clearance &&
              first.witness_upper_m - expected_clearance <= 1.0e-12,
              "point witness encloses the fixed-diagonal plane value");
        check(first.witness_upper_m - first.lower_bound_m >=
                  mandatory_ulp,
              "point lower bound includes the mandatory full float ULP");
        check(double_bits(first.witness.body_x) ==
                  double_bits(static_cast<double>(points[index].x)) &&
              double_bits(first.witness.body_y) ==
                  double_bits(static_cast<double>(points[index].y)) &&
              double_bits(first.witness.body_z) ==
                  double_bits(static_cast<double>(points[index].z)) &&
              double_bits(first.witness.surface_x) ==
                  double_bits(static_cast<double>(points[index].x)) &&
              double_bits(first.witness.surface_y) ==
                  double_bits(expected_heights[index]) &&
              double_bits(first.witness.surface_z) ==
                  double_bits(static_cast<double>(points[index].z)),
              "point witness exposes promoted body and surface geometry");
        check(first.witness.terrain_triangle_index ==
                  expected_triangles[index] &&
              first.witness.primitive_index == 0 &&
              first.witness.cell_x == 0 &&
              first.witness.cell_z == 0 &&
              first.witness.candidate_kind == 3 &&
              first.witness.candidate_subindex == 0 &&
              first.witness.patch_index == 0 &&
              first.witness.segment_parameter == 0.0,
              "point witness identifies the selected fixed triangle");
        check(first.witness.terrain_weight_0 ==
                  expected_weights[index][0] &&
              first.witness.terrain_weight_1 ==
                  expected_weights[index][1] &&
              first.witness.terrain_weight_2 ==
                  expected_weights[index][2],
              "point witness has deterministic weight diagnostics");
        const G1ClearanceWork expected_work = {
            1, 1, 1, 0, 0, 0
        };
        check(clearance_work_same(first.work, expected_work),
              "point query charges one query, cell, and triangle pair");
    }
}

static void require_point_oracle_enclosure(
    const heightfield& field,
    vec3 point,
    long double exact_surface,
    const char* message)
{
    const G1ClearanceResult first = require_point_ok(field, point);
    const G1ClearanceResult second = require_point_ok(field, point);
    const long double exact_clearance =
        static_cast<long double>(point.y) - exact_surface;
    check(clearance_result_same(first, second),
          "point-special proof is deterministic");
    check(static_cast<long double>(first.lower_bound_m) <=
              exact_clearance &&
          exact_clearance <=
              static_cast<long double>(first.witness_upper_m) &&
          first.witness_upper_m - first.lower_bound_m <=
              G1ClearanceMaximumCertificateWidthM,
          message);
    check(double_bits(first.witness.body_x) ==
              double_bits(static_cast<double>(point.x)) &&
          double_bits(first.witness.body_z) ==
              double_bits(static_cast<double>(point.z)) &&
          double_bits(first.witness.surface_x) ==
              double_bits(static_cast<double>(point.x)) &&
          double_bits(first.witness.surface_z) ==
              double_bits(static_cast<double>(point.z)) &&
          first.witness.candidate_kind == 3 &&
          first.witness.segment_parameter == 0.0,
          "point-special diagnostics retain canonical input XZ and t=0");

    G1SurfaceSample producer = {};
    check(g1_surface_query_v2(
              producer, field, point.x, point.z) ==
              G1SurfaceQueryValid,
          "point-special oracle fixture has a valid producer sample");
    const volatile double producer_clearance =
        static_cast<double>(point.y) -
        static_cast<double>(producer.height);
    check(first.lower_bound_m <= producer_clearance,
          "point-special lower bound preserves the producer inequality");

    G1ClearanceBudget insufficient = g1_pose_clearance_budget();
    insufficient.maximum_point_queries = 0;
    G1ClearanceResult failed_output = seeded_result(74.0);
    const ByteSnapshot<G1ClearanceResult> output_before(failed_output);
    const ByteSnapshot<G1ClearanceBudget> limits_before(insufficient);
    char error[128] = {};
    check(g1_point_clearance(
              failed_output, insufficient, field, point,
              error, static_cast<int>(sizeof(error))) ==
              G1ClearanceBudgetExceeded,
          "point-special fixture preflights its point-query budget");
    check(output_before.same(failed_output) &&
          limits_before.same(insufficient),
          "point-special failure is transactional");
}

static void test_point_special_exact_bit_regression()
{
    heightfield field;
    const float cell_size =
        float_from_bits(UINT32_C(0x3a83126f));
    const float coordinate =
        float_from_bits(UINT32_C(0x3727c5ac));
    point_make_field(
        field, 2, 2, 0.0f, 0.0f, cell_size);
    field.heights(0) = 0.0f;
    field.heights(1) =
        float_from_bits(UINT32_C(0x3e800000));
    field.heights(2) =
        float_from_bits(UINT32_C(0x3f000000));
    field.heights(3) =
        float_from_bits(UINT32_C(0x3f400000));
    const vec3 point(coordinate, 1.0f, coordinate);

    heightfield_cell producer_cell = {};
    check(terrain_v2_locate_cell(
              field, point.x, point.z, producer_cell),
          "exact-bit point fixture has a producer cell");
    const volatile double diagnostic_weight_0 =
        1.0 - producer_cell.tx;
    const volatile double diagnostic_weight_1 =
        producer_cell.tx - producer_cell.tz;
    const volatile double diagnostic_weight_2 = producer_cell.tz;
    const long double diagnostic_exact_sum =
        static_cast<long double>(diagnostic_weight_0) +
        static_cast<long double>(diagnostic_weight_1) +
        static_cast<long double>(diagnostic_weight_2);
    check(diagnostic_exact_sum != 1.0L,
          "exact-bit fixture exposes rounded weights that are diagnostics only");

    const long double exact_surface =
        static_cast<long double>(field.heights(3)) *
        static_cast<long double>(coordinate) /
        static_cast<long double>(cell_size);
    require_point_oracle_enclosure(
        field, point, exact_surface,
        "point-special bounds enclose the exact-bit point-plane oracle");
}

static void test_point_rejects_unproved_materialized_diagonal()
{
    heightfield field;
    const float origin =
        float_from_bits(UINT32_C(0x1645c2b2));
    const float cell_size =
        float_from_bits(UINT32_C(0x278eecba));
    point_make_field(
        field, 5, 4, origin, origin, cell_size);
    const vec3 point(
        float_from_bits(UINT32_C(0x287a1e46)),
        1.0f,
        float_from_bits(UINT32_C(0x2832a7e9)));

    check(terrain_heightfield_is_queryable(field),
          "unequal-span point fixture is a queryable G1HF/v2 field");
    heightfield_cell cell = {};
    check(terrain_v2_locate_cell(
              field, point.x, point.z, cell) &&
          cell.x0 == 3 && cell.z0 == 2,
          "unequal-span point fixture selects its exact cell");
    check(double_bits(cell.tx) == UINT64_C(0x3fe00000394b96ce) &&
          double_bits(cell.tz) == UINT64_C(0x3fe00000394b96ce) &&
          cell.tx >= cell.tz,
          "producer fractions tie exactly and select T0");

    const volatile double x0_product =
        static_cast<double>(cell.x0) *
        static_cast<double>(field.cell_size);
    const volatile double x0 =
        static_cast<double>(field.origin_x) + x0_product;
    const volatile double x1_product =
        static_cast<double>(cell.x0 + 1) *
        static_cast<double>(field.cell_size);
    const volatile double x1 =
        static_cast<double>(field.origin_x) + x1_product;
    const volatile double z0_product =
        static_cast<double>(cell.z0) *
        static_cast<double>(field.cell_size);
    const volatile double z0 =
        static_cast<double>(field.origin_z) + z0_product;
    const volatile double z1_product =
        static_cast<double>(cell.z0 + 1) *
        static_cast<double>(field.cell_size);
    const volatile double z1 =
        static_cast<double>(field.origin_z) + z1_product;
    const volatile double span_x = x1 - x0;
    const volatile double span_z = z1 - z0;
    const volatile double numerator_x =
        static_cast<double>(point.x) - x0;
    const volatile double numerator_z =
        static_cast<double>(point.z) - z0;
    check(double_bits(span_x) == UINT64_C(0x3cf1dd9740000002) &&
          double_bits(span_z) == UINT64_C(0x3cf1dd9740000000) &&
          span_x > span_z,
          "authoritative materialized axis spans are unequal");
    check(double_bits(numerator_x) ==
              UINT64_C(0x3ce1dd977ff9d1ec) &&
          double_bits(numerator_z) ==
              UINT64_C(0x3ce1dd977ff9d1ec) &&
          numerator_x > 0.0,
          "authoritative point numerators are the same positive dyadic");
    // The identical positive numerator divided by the strictly larger X
    // span proves exact-real tx < tz, so the materialized triangle is T1.
    // The rounded producer fractions above instead tie and choose T0.

    const int offset = cell.z0 * field.nx + cell.x0;
    field.heights(offset) = -8.0f;
    field.heights(offset + 1) = 8.0f;
    field.heights(offset + field.nx) = 8.0f;
    field.heights(offset + field.nx + 1) = -8.0f;

    G1ClearanceResult output = seeded_result(72.0);
    const ByteSnapshot<G1ClearanceResult> output_before(output);
    const G1ClearanceBudget limits = g1_pose_clearance_budget();
    char error[128] = {};
    check(g1_point_clearance(
              output, limits, field, point,
              error, static_cast<int>(sizeof(error))) ==
              G1ClearanceUncertified,
          "point fails closed when the authoritative diagonal is unproved");
    check(output_before.same(output),
          "unproved materialized diagonal preserves point output");
}

static void test_point_rejects_false_rounded_diagonal_tie()
{
    heightfield field;
    point_make_field(
        field, 2, 2,
        float_from_bits(UINT32_C(0x3dcccccd)),
        float_from_bits(UINT32_C(0x3d4ccccd)),
        float_from_bits(UINT32_C(0x5921729f)));
    const float coordinate =
        float_from_bits(UINT32_C(0x58a1729f));
    const vec3 point(coordinate, 1.0f, coordinate);
    field.heights(0) = -8.0f;
    field.heights(1) = 8.0f;
    field.heights(2) = 8.0f;
    field.heights(3) = -8.0f;

    check(terrain_heightfield_is_queryable(field),
          "false-rounded-tie fixture is a queryable G1HF/v2 field");
    heightfield_cell cell = {};
    check(terrain_v2_locate_cell(
              field, point.x, point.z, cell) &&
          double_bits(cell.tx) == UINT64_C(0x3fe0000000000000) &&
          double_bits(cell.tz) == UINT64_C(0x3fe0000000000000),
          "false-rounded-tie producer fractions tie at one half");

    const volatile double x1_product =
        static_cast<double>(field.cell_size);
    const volatile double x1 =
        static_cast<double>(field.origin_x) + x1_product;
    const volatile double z1_product =
        static_cast<double>(field.cell_size);
    const volatile double z1 =
        static_cast<double>(field.origin_z) + z1_product;
    const double x0 = static_cast<double>(field.origin_x);
    const double z0 = static_cast<double>(field.origin_z);
    const volatile double numerator_x =
        static_cast<double>(point.x) - x0;
    const volatile double numerator_z =
        static_cast<double>(point.z) - z0;
    const volatile double span_x = x1 - x0;
    const volatile double span_z = z1 - z0;
    check(x0 > z0 && double_bits(x1) == double_bits(z1) &&
          double_bits(static_cast<double>(point.x)) ==
              double_bits(static_cast<double>(point.z)),
          "false-rounded-tie source operands are distinct on X and Z");
    check(double_bits(numerator_x) ==
              UINT64_C(0x43142e53e0000000) &&
          double_bits(numerator_z) ==
              UINT64_C(0x43142e53e0000000) &&
          double_bits(span_x) == UINT64_C(0x43242e53e0000000) &&
          double_bits(span_z) == UINT64_C(0x43242e53e0000000),
          "distinct exact ratios have identical rounded differences");
    // With a shared point p and upper node n, 0 < z0 < x0 < p < n,
    // (p-o)/(n-o) is strictly decreasing in o.  Therefore exact-real
    // tx < tz even though all four rounded differences above are equal.

    G1ClearanceResult output = seeded_result(76.0);
    const ByteSnapshot<G1ClearanceResult> output_before(output);
    const G1ClearanceBudget limits = g1_pose_clearance_budget();
    char error[128] = {};
    check(g1_point_clearance(
              output, limits, field, point,
              error, static_cast<int>(sizeof(error))) ==
              G1ClearanceUncertified,
          "rounded difference equality cannot certify a diagonal tie");
    check(output_before.same(output),
          "false rounded diagonal tie preserves point output");
}

static void test_point_domain_boundaries()
{
    heightfield exact_field;
    point_make_field(exact_field, 2, 2, 0.5f, 0.5f, 1.0f);
    require_point_ok(exact_field, vec3(0.5f, 1.0f, 0.5f));
    require_point_ok(exact_field, vec3(1.5f, 1.0f, 1.5f));

    const float below_minimum =
        float_from_bits(float_bits(0.5f) - 1);
    const float above_maximum =
        float_from_bits(float_bits(1.5f) + 1);
    require_point_failure(
        exact_field, vec3(below_minimum, 1.0f, 1.0f),
        G1ClearanceOutsideDomain,
        "one-ULP excursion below minimum X is outside");
    require_point_failure(
        exact_field, vec3(1.0f, 1.0f, below_minimum),
        G1ClearanceOutsideDomain,
        "one-ULP excursion below minimum Z is outside");
    require_point_failure(
        exact_field, vec3(above_maximum, 1.0f, 1.0f),
        G1ClearanceOutsideDomain,
        "one-ULP excursion above maximum X is outside");
    require_point_failure(
        exact_field, vec3(1.0f, 1.0f, above_maximum),
        G1ClearanceOutsideDomain,
        "one-ULP excursion above maximum Z is outside");

    heightfield nonrepresentable_maximum;
    point_make_field(
        nonrepresentable_maximum, 3, 2,
        float_from_bits(UINT32_C(0x3dcccccd)),
        float_from_bits(UINT32_C(0x3dcccccd)),
        float_from_bits(UINT32_C(0x3e4ccccd)));
    check(terrain_heightfield_is_queryable(nonrepresentable_maximum),
          "non-binary32 maximum fixture is structurally valid");
    const volatile double maximum_product =
        2.0 * static_cast<double>(nonrepresentable_maximum.cell_size);
    const volatile double maximum_x =
        static_cast<double>(nonrepresentable_maximum.origin_x) +
        maximum_product;
    check(maximum_x != static_cast<double>(0.5f),
          "authoritative maximum node is not binary32 representable");
    const vec3 inward(0.5f, 1.0f,
                      nonrepresentable_maximum.origin_z);
    require_point_ok(nonrepresentable_maximum, inward);
    require_point_failure(
        nonrepresentable_maximum,
        vec3(float_from_bits(UINT32_C(0x3f000001)),
             1.0f, nonrepresentable_maximum.origin_z),
        G1ClearanceOutsideDomain,
        "next binary32 value beyond nonrepresentable maximum is outside");

    heightfield zero_origin;
    point_make_field(zero_origin);
    const G1ClearanceResult canonical = require_point_ok(
        zero_origin, vec3(-0.0f, 1.0f, -0.0f));
    check(double_bits(canonical.witness.body_x) == 0 &&
          double_bits(canonical.witness.body_z) == 0 &&
          double_bits(canonical.witness.surface_x) == 0 &&
          double_bits(canonical.witness.surface_z) == 0,
          "point query canonicalizes signed-zero XZ to positive zero");
}

static void test_point_input_and_field_rejection()
{
    heightfield field;
    point_make_field(field);
    const float hostile_values[] = {
        float_from_bits(UINT32_C(0x00000001)),
        float_from_bits(UINT32_C(0x80000001)),
        float_from_bits(UINT32_C(0x7fc00001)),
        std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity()
    };
    for (size_t value_index = 0;
         value_index < sizeof(hostile_values) / sizeof(hostile_values[0]);
         ++value_index) {
        for (int component = 0; component < 3; ++component) {
            vec3 point(0.5f, 1.0f, 0.5f);
            if (component == 0) point.x = hostile_values[value_index];
            if (component == 1) point.y = hostile_values[value_index];
            if (component == 2) point.z = hostile_values[value_index];
            require_point_failure(
                field, point, G1ClearanceInvalidInput,
                "non-runtime XYZ component is invalid point input");
        }
    }

    field.version = 1;
    require_point_failure(
        field, vec3(0.5f, 1.0f, 0.5f),
        G1ClearanceInvalidField,
        "G1HF/v1 is invalid for certified point clearance");
    field.version = 2;

    const int saved_size = field.heights.size;
    field.heights.size = saved_size - 1;
    require_point_failure(
        field, vec3(0.5f, 1.0f, 0.5f),
        G1ClearanceInvalidField,
        "wrong height storage shape is invalid");
    field.heights.size = saved_size;

    float* const saved_data = field.heights.data;
    field.heights.data = NULL;
    require_point_failure(
        field, vec3(0.5f, 1.0f, 0.5f),
        G1ClearanceInvalidField,
        "null height storage is invalid");
    field.heights.data = saved_data;

    const uint32_t invalid_height_bits[] = {
        UINT32_C(0x7fc00001),
        UINT32_C(0x7f800000),
        UINT32_C(0x00000001),
        UINT32_C(0x80000000)
    };
    const char* const invalid_height_messages[] = {
        "NaN visited triangle height fails closed",
        "infinite visited triangle height fails closed",
        "subnormal visited triangle height fails closed",
        "negative-zero visited triangle height fails closed"
    };
    for (size_t index = 0;
         index < sizeof(invalid_height_bits) /
                     sizeof(invalid_height_bits[0]);
         ++index) {
        field.heights.set(0.0f);
        std::memcpy(
            &field.heights(3), &invalid_height_bits[index],
            sizeof(invalid_height_bits[index]));
        check(float_bits(field.heights(3)) ==
                  invalid_height_bits[index],
              "hostile height fixture preserves object representation");
        require_point_failure(
            field, vec3(0.75f, 1.0f, 0.25f),
            G1ClearanceInvalidField,
            invalid_height_messages[index]);
    }
}

static void test_point_exterior_height_independence()
{
    heightfield first;
    heightfield second;
    point_make_field(first, 2, 2, 0.0f, 0.0f, 1.0f, -10.0f);
    point_make_field(second, 2, 2, 0.0f, 0.0f, 1.0f, 25.0f);
    const float heights[4] = {0.0f, 2.0f, 4.0f, 10.0f};
    for (int index = 0; index < 4; ++index) {
        first.heights(index) = heights[index];
        second.heights(index) = heights[index];
    }
    const vec3 inside(0.25f, 20.0f, 0.75f);
    const G1ClearanceResult first_result = require_point_ok(first, inside);
    const G1ClearanceResult second_result = require_point_ok(second, inside);
    check(clearance_result_same(first_result, second_result),
          "in-domain point result ignores exterior height bits");

    const vec3 outside(
        float_from_bits(float_bits(1.0f) + 1), 20.0f, 0.5f);
    G1ClearanceResult first_output = seeded_result(75.0);
    G1ClearanceResult second_output = first_output;
    const ByteSnapshot<G1ClearanceResult> first_before(first_output);
    const ByteSnapshot<G1ClearanceResult> second_before(second_output);
    const G1ClearanceBudget limits = g1_pose_clearance_budget();
    char first_error[128] = {};
    char second_error[128] = {};
    const G1ClearanceStatus first_status = g1_point_clearance(
        first_output, limits, first, outside,
        first_error, static_cast<int>(sizeof(first_error)));
    const G1ClearanceStatus second_status = g1_point_clearance(
        second_output, limits, second, outside,
        second_error, static_cast<int>(sizeof(second_error)));
    check(first_status == G1ClearanceOutsideDomain &&
          second_status == G1ClearanceOutsideDomain,
          "outside status is independent of exterior height");
    check(first_before.same(first_output) &&
          second_before.same(second_output),
          "outside result remains transactional for both exteriors");
}

static void test_point_budget_preflight_and_aliases()
{
    heightfield field;
    point_make_field(field);
    const vec3 point(0.5f, 1.0f, 0.5f);

    G1ClearanceBudget minimum = {};
    minimum.maximum_point_queries = 1;
    minimum.maximum_cells = 1;
    minimum.maximum_primitive_triangle_pairs = 1;
    const G1ClearanceResult minimum_result =
        require_point_ok(field, point, minimum);
    const G1ClearanceWork expected_work = {1, 1, 1, 0, 0, 0};
    check(clearance_work_same(minimum_result.work, expected_work),
          "tight point budget reports exact deterministic work");

    const BudgetMember required_fields[] = {
        &G1ClearanceBudget::maximum_point_queries,
        &G1ClearanceBudget::maximum_cells,
        &G1ClearanceBudget::maximum_primitive_triangle_pairs
    };
    for (size_t index = 0;
         index < sizeof(required_fields) / sizeof(required_fields[0]);
         ++index) {
        G1ClearanceBudget insufficient = minimum;
        insufficient.*required_fields[index] = 0;
        require_point_failure(
            field, point, insufficient,
            G1ClearanceBudgetExceeded,
            "point preflight rejects a tightened required work unit");
    }

    heightfield poisoned;
    point_make_field(poisoned);
    poisoned.heights(3) =
        float_from_bits(UINT32_C(0x7fc00001));
    G1ClearanceBudget no_cells = minimum;
    no_cells.maximum_cells = 0;
    require_point_failure(
        poisoned, point, no_cells,
        G1ClearanceBudgetExceeded,
        "point work preflight occurs before the first terrain height load");

    G1ClearanceBudget aliased_limits = minimum;
    aliased_limits.maximum_point_queries = 0;
    G1ClearanceResult output = seeded_result(77.0);
    const ByteSnapshot<G1ClearanceResult> output_before(output);
    const ByteSnapshot<G1ClearanceBudget> limits_before(aliased_limits);
    const G1ClearanceStatus limit_alias_status = g1_point_clearance(
        output, aliased_limits, field, point,
        reinterpret_cast<char*>(&aliased_limits),
        static_cast<int>(sizeof(aliased_limits)));
    check(limit_alias_status == G1ClearanceBudgetExceeded,
          "aliased limit diagnostic preserves budget status");
    check(output_before.same(output) &&
          limits_before.same(aliased_limits),
          "budget failure preserves output and aliased limits");

    G1ClearanceBudget insufficient = minimum;
    insufficient.maximum_point_queries = 0;
    output = seeded_result(79.0);
    const ByteSnapshot<G1ClearanceResult> aliased_output_before(output);
    const G1ClearanceStatus output_alias_status = g1_point_clearance(
        output, insufficient, field, point,
        reinterpret_cast<char*>(&output),
        static_cast<int>(sizeof(output)));
    check(output_alias_status == G1ClearanceBudgetExceeded,
          "aliased output diagnostic preserves budget status");
    check(aliased_output_before.same(output),
          "budget failure preserves aliased point output");
}

static void require_point_guard_width(
    const heightfield& field,
    vec3 point,
    double minimum_guard,
    const char* message)
{
    const G1ClearanceResult result = require_point_ok(field, point);
    check(result.witness_upper_m - result.lower_bound_m >=
              minimum_guard,
          message);
}

static void test_point_float_output_guards()
{
    heightfield upward;
    point_make_field(upward);
    const float one_up =
        float_from_bits(float_bits(1.0f) + 1);
    upward.heights(0) = 1.0f;
    upward.heights(1) = one_up;
    upward.heights(2) = 1.0f;
    upward.heights(3) = one_up;
    const vec3 upward_point(0.75f, 2.0f, 0.0f);
    G1SurfaceSample upward_sample = {};
    check(g1_surface_query_v2(
              upward_sample, upward,
              upward_point.x, upward_point.z) ==
              G1SurfaceQueryValid &&
          float_bits(upward_sample.height) == float_bits(one_up),
          "producer fixture forces upward binary32 height rounding");
    const double one_ulp =
        static_cast<double>(one_up) - 1.0;
    const G1ClearanceResult upward_result =
        require_point_ok(upward, upward_point);
    const double upward_continuous_height =
        1.0 + 0.75 * one_ulp;
    check(upward_result.witness.surface_y >=
              upward_continuous_height - 1.0e-15 &&
          upward_result.witness.surface_y <=
              upward_continuous_height + 1.0e-15,
          "point witness remains on the continuous affine triangle");
    check(upward_result.witness_upper_m -
              upward_result.lower_bound_m >= one_ulp,
          "upward-rounded sample retains a full binary32 ULP guard");

    const float minimum_normal =
        std::numeric_limits<float>::min();
    heightfield negative_zero_band;
    point_make_field(negative_zero_band);
    negative_zero_band.heights(0) = -minimum_normal;
    negative_zero_band.heights(1) = 0.0f;
    negative_zero_band.heights(2) = -minimum_normal;
    negative_zero_band.heights(3) = 0.0f;
    const vec3 zero_band_point(0.5f, 0.0f, 0.0f);
    G1SurfaceSample negative_sample = {};
    check(g1_surface_query_v2(
              negative_sample, negative_zero_band,
              zero_band_point.x, zero_band_point.z) ==
              G1SurfaceQueryValid &&
          float_bits(negative_sample.height) == 0,
          "negative subnormal producer result canonicalizes to +zero");
    require_point_guard_width(
        negative_zero_band, zero_band_point,
        static_cast<double>(minimum_normal),
        "negative zero-band result uses at least FLT_MIN guard");

    heightfield positive_zero_band;
    point_make_field(positive_zero_band);
    positive_zero_band.heights(0) = minimum_normal;
    positive_zero_band.heights(1) = 0.0f;
    positive_zero_band.heights(2) = minimum_normal;
    positive_zero_band.heights(3) = 0.0f;
    G1SurfaceSample positive_sample = {};
    check(g1_surface_query_v2(
              positive_sample, positive_zero_band,
              zero_band_point.x, zero_band_point.z) ==
              G1SurfaceQueryValid &&
          float_bits(positive_sample.height) == 0,
          "positive subnormal producer result canonicalizes to +zero");
    require_point_guard_width(
        positive_zero_band, zero_band_point,
        static_cast<double>(minimum_normal),
        "positive zero-band result uses at least FLT_MIN guard");

    const float binades[] = {1.0f, 2.0f};
    const vec3 side_points[] = {
        vec3(0.75f, 4.0f, 0.25f),
        vec3(0.25f, 4.0f, 0.75f)
    };
    for (size_t binade_index = 0;
         binade_index < sizeof(binades) / sizeof(binades[0]);
         ++binade_index) {
        const float boundary = binades[binade_index];
        const float below =
            float_from_bits(float_bits(boundary) - 1);
        const float above =
            float_from_bits(float_bits(boundary) + 1);
        const double larger_ulp =
            static_cast<double>(above) -
            static_cast<double>(boundary);
        heightfield crossing;
        point_make_field(crossing);
        crossing.heights(0) = below;
        crossing.heights(1) = boundary;
        crossing.heights(2) = boundary;
        crossing.heights(3) = boundary;
        for (size_t side = 0;
             side < sizeof(side_points) / sizeof(side_points[0]);
             ++side) {
            require_point_guard_width(
                crossing, side_points[side], larger_ulp,
                "both fixed triangles use larger adjacent binade ULP");
        }
    }

    heightfield flat_sixteen;
    point_make_field(flat_sixteen);
    flat_sixteen.heights.set(16.0f);
    const vec3 flat_point(0.5f, 20.0f, 0.5f);
    G1SurfaceSample flat_sample = {};
    check(g1_surface_query_v2(
              flat_sample, flat_sixteen,
              flat_point.x, flat_point.z) ==
              G1SurfaceQueryValid,
          "flat-16 producer fixture is valid");
    const double flat_ulp =
        static_cast<double>(
            float_from_bits(float_bits(16.0f) + 1)) - 16.0;
    check(flat_ulp > G1ClearanceMaximumCertificateWidthM,
          "flat-16 full float ULP exceeds certificate cap");
    G1ClearanceResult output = seeded_result(81.0);
    const ByteSnapshot<G1ClearanceResult> output_before(output);
    const G1ClearanceBudget limits = g1_pose_clearance_budget();
    char error[128] = {};
    check(g1_point_clearance(
              output, limits, flat_sixteen, flat_point,
              error, static_cast<int>(sizeof(error))) ==
              G1ClearanceUncertified,
          "flat-16 point fails closed on mandatory guard width");
    check(output_before.same(output),
          "flat-16 uncertified point preserves output");

    output = seeded_result(83.0);
    const ByteSnapshot<G1ClearanceResult> aliased_before(output);
    check(g1_point_clearance(
              output, limits, flat_sixteen, flat_point,
              reinterpret_cast<char*>(&output),
              static_cast<int>(sizeof(output))) ==
              G1ClearanceUncertified,
          "flat-16 aliased diagnostic preserves semantic status");
    check(aliased_before.same(output),
          "flat-16 aliased diagnostic preserves output bytes");
}

static void task34_make_smooth_plane_field(heightfield& field)
{
    const float cell = float_from_bits(UINT32_C(0x3ca3d70a));
    const float delta = float_from_bits(UINT32_C(0x3d0de3b8));
    point_make_field(field, 9, 9, 0.0f, 0.0f, cell);
    for (int z = 0; z < field.nz; ++z) {
        for (int x = 0; x < field.nx; ++x) {
            const volatile float height =
                static_cast<float>(x) * delta;
            check(static_cast<double>(height) ==
                      static_cast<double>(x) *
                          static_cast<double>(delta),
                  "smooth-plane fixture stores exact column products");
            field.heights(x + z * field.nx) = height;
        }
    }
}

static void test_task34_smooth_plane_sphere_false_clear()
{
    heightfield field;
    task34_make_smooth_plane_field(field);
    const float cell = field.cell_size;
    const float radius = float_from_bits(UINT32_C(0x3ca3d70a));
    const float center_x = 4.0f * cell;
    const float center_z = 4.0f * cell;
    const float surface = field.heights(4);
    const volatile float center_y = surface + 0.039f;
    const vec3 center(center_x, center_y, center_z);

    const long double slope =
        static_cast<long double>(field.heights(1)) /
        static_cast<long double>(cell);
    const long double oracle =
        static_cast<long double>(center.y) -
        static_cast<long double>(surface) -
        static_cast<long double>(radius) *
            std::sqrt(1.0L + slope * slope);
    check(oracle < 0.0L &&
              std::fabs(oracle + 0.001L) < 2.0e-6L,
          "smooth-plane sphere fixture has the intended negative oracle");

    G1ClearanceResult output = seeded_result(131.0);
    char error[256] = {};
    const G1ClearanceStatus status = g1_sphere_clearance(
        output, g1_pose_clearance_budget(), field,
        center, radius, error, static_cast<int>(sizeof(error)));
    check(status == G1ClearanceOk,
          "smooth-plane sphere must produce a certificate");
    check(static_cast<long double>(output.lower_bound_m) <= oracle &&
              oracle <=
                  static_cast<long double>(output.witness_upper_m) &&
              output.witness_upper_m - output.lower_bound_m <=
                  G1ClearanceMaximumCertificateWidthM,
          "smooth-plane sphere certificate encloses the analytic oracle");
    check(output.witness_upper_m < 0.0,
          "smooth-plane sphere detects the false-clear collision");
}

static void test_task34_finite_capsule_plane_and_reversal()
{
    heightfield field;
    point_make_field(field, 9, 9, 0.0f, 0.0f, 0.25f);
    const double slope_x = 0.5;
    const double slope_z = -0.25;
    const double intercept = 0.125;
    for (int z = 0; z < field.nz; ++z) {
        for (int x = 0; x < field.nx; ++x) {
            const double px = static_cast<double>(x) * 0.25;
            const double pz = static_cast<double>(z) * 0.25;
            field.heights(x + z * field.nx) = static_cast<float>(
                slope_x * px + slope_z * pz + intercept);
        }
    }

    const vec3 endpoint_a(0.5f, 1.0f, 0.5f);
    const vec3 endpoint_b(1.5f, 0.75f, 1.25f);
    const float radius = 0.125f;
    const double plane_a =
        static_cast<double>(endpoint_a.y) -
        slope_x * static_cast<double>(endpoint_a.x) -
        slope_z * static_cast<double>(endpoint_a.z) - intercept;
    const double plane_b =
        static_cast<double>(endpoint_b.y) -
        slope_x * static_cast<double>(endpoint_b.x) -
        slope_z * static_cast<double>(endpoint_b.z) - intercept;
    const double oracle = std::fmin(plane_a, plane_b) -
        static_cast<double>(radius) *
            std::sqrt(1.0 + slope_x * slope_x + slope_z * slope_z);

    G1ClearanceResult forward = seeded_result(137.0);
    G1ClearanceResult reverse = seeded_result(139.0);
    char forward_error[256] = {};
    char reverse_error[256] = {};
    const G1ClearanceStatus forward_status = g1_capsule_clearance(
        forward, g1_pose_clearance_budget(), field,
        endpoint_a, endpoint_b, radius,
        forward_error, static_cast<int>(sizeof(forward_error)));
    const G1ClearanceStatus reverse_status = g1_capsule_clearance(
        reverse, g1_pose_clearance_budget(), field,
        endpoint_b, endpoint_a, radius,
        reverse_error, static_cast<int>(sizeof(reverse_error)));
    check(forward_status == G1ClearanceOk &&
              reverse_status == G1ClearanceOk,
          "finite planar capsule and its reversal must certify");
    check(forward.lower_bound_m <= oracle &&
              oracle <= forward.witness_upper_m &&
              forward.witness_upper_m - forward.lower_bound_m <=
                  G1ClearanceMaximumCertificateWidthM,
          "finite planar capsule certificate encloses its oracle");
    check(clearance_result_same(forward, reverse),
          "capsule endpoint reversal is bit-identical");
}

static void test_task34_footprint_domain_budget_and_transaction()
{
    heightfield field;
    point_make_field(field, 5, 5, 0.0f, 0.0f, 0.25f);
    const float radius = 0.25f;
    const vec3 minimum_tangent(0.25f, 1.0f, 0.25f);
    const vec3 maximum_tangent(0.75f, 1.0f, 0.75f);

    G1ClearanceResult minimum_output = seeded_result(149.0);
    G1ClearanceResult maximum_output = seeded_result(151.0);
    check(g1_sphere_clearance(
              minimum_output, g1_pose_clearance_budget(), field,
              minimum_tangent, radius, NULL, 0) == G1ClearanceOk,
          "sphere footprint exactly tangent to minimum XZ certifies");
    check(g1_sphere_clearance(
              maximum_output, g1_pose_clearance_budget(), field,
              maximum_tangent, radius, NULL, 0) == G1ClearanceOk,
          "sphere footprint exactly tangent to maximum XZ certifies");

    const vec3 outside(
        std::nextafter(minimum_tangent.x, 0.0f),
        minimum_tangent.y,
        minimum_tangent.z);
    G1ClearanceResult outside_output = seeded_result(157.0);
    const ByteSnapshot<G1ClearanceResult> outside_before(outside_output);
    check(g1_sphere_clearance(
              outside_output, g1_pose_clearance_budget(), field,
              outside, radius, NULL, 0) == G1ClearanceOutsideDomain,
          "one-ULP footprint excursion is outside-domain");
    check(outside_before.same(outside_output),
          "outside-domain sphere failure is transactional");

    heightfield poisoned = field;
    poisoned.heights.set(std::numeric_limits<float>::quiet_NaN());
    G1ClearanceBudget no_cells = g1_pose_clearance_budget();
    no_cells.maximum_cells = 0;
    G1ClearanceResult budget_output = seeded_result(163.0);
    const ByteSnapshot<G1ClearanceResult> budget_before(budget_output);
    check(g1_sphere_clearance(
              budget_output, no_cells, poisoned,
              vec3(0.5f, 1.0f, 0.5f), radius,
              NULL, 0) == G1ClearanceBudgetExceeded,
          "sphere work budget is rejected before the first height load");
    check(budget_before.same(budget_output),
          "budget-exceeded sphere failure is transactional");
}

class RoundingModeGuard
{
public:
    RoundingModeGuard() : saved_(std::fegetround()) {}
    ~RoundingModeGuard() { std::fesetround(saved_); }
    int saved() const { return saved_; }

private:
    int saved_;
};

#if defined(__SSE__) || defined(_M_X64) || defined(_M_IX86_FP)
class MxcsrGuard
{
public:
    MxcsrGuard() : saved_(_mm_getcsr()) {}
    ~MxcsrGuard() { _mm_setcsr(saved_); }
    uint32_t saved() const { return saved_; }

private:
    uint32_t saved_;
};
#endif

static void require_disjoint_diagnostic(
    const char* diagnostic,
    size_t capacity)
{
    check(capacity > 1 && diagnostic[0] != 'x' &&
          std::memchr(diagnostic, '\0', capacity) != NULL,
          "disjoint diagnostic is written and terminated");
}

struct ArithmeticEnvironmentSnapshot
{
    int rounding_mode;
#if defined(__SSE__) || defined(_M_X64) || defined(_M_IX86_FP)
    uint32_t mxcsr;
#endif

    ArithmeticEnvironmentSnapshot()
        : rounding_mode(std::fegetround())
#if defined(__SSE__) || defined(_M_X64) || defined(_M_IX86_FP)
        , mxcsr(_mm_getcsr())
#endif
    {
    }

    bool same() const
    {
        if (std::fegetround() != rounding_mode) {
            return false;
        }
#if defined(__SSE__) || defined(_M_X64) || defined(_M_IX86_FP)
        if (_mm_getcsr() != mxcsr) {
            return false;
        }
#endif
        return true;
    }
};

static void test_output_diagnostic_aliases_in_current_environment()
{
    for (int raw_entry = PublicLift;
         raw_entry < PublicStatusEntryCount;
         ++raw_entry) {
        const PublicStatusEntry entry =
            static_cast<PublicStatusEntry>(raw_entry);
        const G1ClearanceBudget limits = public_entry_budget(entry);

        PublicOutputs baseline_outputs = seeded_public_outputs();
        char diagnostic[64];
        std::memset(diagnostic, 'x', sizeof(diagnostic));
        const G1ClearanceStatus baseline_status =
            invoke_public_status_entry(
                entry, baseline_outputs, limits,
                diagnostic, static_cast<int>(sizeof(diagnostic)));
        check(baseline_status != G1ClearanceOk,
              "alias fixture reaches a diagnostic status");
        require_disjoint_diagnostic(diagnostic, sizeof(diagnostic));

        for (int overlap_kind = 0; overlap_kind < 2;
             ++overlap_kind) {
            PublicOutputs outputs = seeded_public_outputs();
            const ByteSnapshot<PublicOutputs> before(outputs);
            const PublicProtectedSpan span =
                public_output_span(entry, outputs);
            check(span.data != NULL && span.size > 2,
                  "public output exposes a nonempty protected span");
            char* const aliased_error = overlap_kind == 0
                ? span.data
                : span.data + 1;
            const int aliased_capacity = overlap_kind == 0
                ? span.size
                : 2;
            const ArithmeticEnvironmentSnapshot environment_before;
            const G1ClearanceStatus status =
                invoke_public_status_entry(
                    entry, outputs, limits,
                    aliased_error, aliased_capacity);
            check(environment_before.same(),
                  "output alias preserves arithmetic environment exactly");
            check(status == baseline_status,
                  "output alias preserves semantic status");
            check(before.same(outputs),
                  "exact or partial diagnostic overlap preserves output");
        }
    }
}

static void test_limit_diagnostic_aliases_in_current_environment(
    bool exceed_factory)
{
    for (int raw_entry = PublicPoint;
         raw_entry < PublicStatusEntryCount;
         ++raw_entry) {
        const PublicStatusEntry entry =
            static_cast<PublicStatusEntry>(raw_entry);
        G1ClearanceBudget fixture = public_entry_budget(entry);
        if (exceed_factory) {
            ++fixture.maximum_cells;
        }

        PublicOutputs baseline_outputs = seeded_public_outputs();
        G1ClearanceBudget baseline_limits = fixture;
        char diagnostic[64];
        std::memset(diagnostic, 'x', sizeof(diagnostic));
        const G1ClearanceStatus baseline_status =
            invoke_public_status_entry(
                entry, baseline_outputs, baseline_limits,
                diagnostic, static_cast<int>(sizeof(diagnostic)));
        check(baseline_status != G1ClearanceOk,
              "limit alias fixture reaches a diagnostic status");
        require_disjoint_diagnostic(diagnostic, sizeof(diagnostic));

        for (int overlap_kind = 0; overlap_kind < 2;
             ++overlap_kind) {
            PublicOutputs outputs = seeded_public_outputs();
            G1ClearanceBudget limits = fixture;
            const ByteSnapshot<PublicOutputs> output_before(outputs);
            const ByteSnapshot<G1ClearanceBudget> limits_before(limits);
            char* const limits_bytes =
                reinterpret_cast<char*>(&limits);
            char* const aliased_error = overlap_kind == 0
                ? limits_bytes
                : limits_bytes + 1;
            const int aliased_capacity = overlap_kind == 0
                ? static_cast<int>(sizeof(limits))
                : 2;
            const ArithmeticEnvironmentSnapshot environment_before;
            const G1ClearanceStatus status =
                invoke_public_status_entry(
                    entry, outputs, limits,
                    aliased_error, aliased_capacity);
            check(environment_before.same(),
                  "limit alias preserves arithmetic environment exactly");
            check(status == baseline_status,
                  "limit alias preserves semantic status");
            check(output_before.same(outputs),
                  "limit alias never changes the public output");
            check(limits_before.same(limits),
                  "exact or partial diagnostic overlap preserves limits");
        }
    }
}

static G1SwingHistory seeded_swing_history()
{
    G1SwingHistory history = {};
    history.initialized = true;
    for (int index = 0; index < 4; ++index) {
        history.previous_sphere_centers[index] = vec3(
            0.1f * static_cast<float>(index + 1),
            0.2f * static_cast<float>(index + 1),
            0.3f * static_cast<float>(index + 1));
    }
    return history;
}

static G1ClearanceStatus invoke_swing_with_history(
    PublicOutputs& outputs,
    const G1ClearanceBudget& limits,
    const G1SwingHistory& history,
    char* error,
    int error_capacity)
{
    heightfield field;
    const G1LegConfig config = g1_left_leg_config();
    const vec3 final_centers[4] = {
        vec3(0.1f, 0.2f, 0.3f),
        vec3(0.2f, 0.3f, 0.4f),
        vec3(0.3f, 0.4f, 0.5f),
        vec3(0.4f, 0.5f, 0.6f)
    };
    return g1_swing_clearance_validate(
        outputs.swing, limits, history, field, config,
        final_centers, false, 0.04f, error, error_capacity);
}

static void test_history_diagnostic_aliases_in_current_environment()
{
    const G1ClearanceBudget limits =
        g1_swing_foot_clearance_budget();
    PublicOutputs baseline_outputs = seeded_public_outputs();
    const G1SwingHistory baseline_history = seeded_swing_history();
    char diagnostic[64];
    std::memset(diagnostic, 'x', sizeof(diagnostic));
    const G1ClearanceStatus baseline_status =
        invoke_swing_with_history(
            baseline_outputs, limits, baseline_history,
            diagnostic, static_cast<int>(sizeof(diagnostic)));
    check(baseline_status != G1ClearanceOk,
          "history alias fixture reaches a diagnostic status");
    require_disjoint_diagnostic(diagnostic, sizeof(diagnostic));

    for (int overlap_kind = 0; overlap_kind < 2;
         ++overlap_kind) {
        PublicOutputs outputs = seeded_public_outputs();
        G1SwingHistory history = seeded_swing_history();
        const ByteSnapshot<PublicOutputs> output_before(outputs);
        const ByteSnapshot<G1SwingHistory> history_before(history);
        char* const history_bytes = reinterpret_cast<char*>(&history);
        char* const aliased_error = overlap_kind == 0
            ? history_bytes
            : history_bytes + 1;
        const int aliased_capacity = overlap_kind == 0
            ? static_cast<int>(sizeof(history))
            : 2;
        const ArithmeticEnvironmentSnapshot environment_before;
        const G1ClearanceStatus status = invoke_swing_with_history(
            outputs, limits, history,
            aliased_error, aliased_capacity);
        check(environment_before.same(),
              "history alias preserves arithmetic environment exactly");
        check(status == baseline_status,
              "history alias preserves semantic status");
        check(output_before.same(outputs),
              "history alias never changes swing output");
        check(history_before.same(history),
              "exact or partial diagnostic overlap preserves history");
    }
}

struct ScalarPrefixEnvelope
{
    unsigned char prefix[sizeof(float)];
    float output;
    unsigned char suffix[2];
};

static_assert(
    offsetof(ScalarPrefixEnvelope, output) == sizeof(float),
    "scalar overlap envelope has adjacent prefix storage");

static void test_diagnostic_range_crossing_into_output()
{
    ScalarPrefixEnvelope envelope = {};
    std::memset(envelope.prefix, 0xa5, sizeof(envelope.prefix));
    envelope.output = float_from_bits(UINT32_C(0x41234567));
    std::memset(envelope.suffix, 0x5a, sizeof(envelope.suffix));
    const ByteSnapshot<ScalarPrefixEnvelope> before(envelope);
    char disjoint[64] = {};
    float baseline_output = envelope.output;
    const float hostile = float_from_bits(UINT32_C(0x7fc00001));
    const G1ClearanceStatus baseline_status = g1_apply_swing_lift_y(
        baseline_output, hostile, hostile,
        disjoint, static_cast<int>(sizeof(disjoint)));
    char* const crossing_error =
        reinterpret_cast<char*>(&envelope.output) - 1;
    const ArithmeticEnvironmentSnapshot environment_before;
    const G1ClearanceStatus status = g1_apply_swing_lift_y(
        envelope.output, hostile, hostile, crossing_error, 2);
    check(environment_before.same(),
          "crossing diagnostic preserves arithmetic environment exactly");
    check(status == baseline_status,
          "crossing diagnostic range preserves semantic status");
    check(before.same(envelope),
          "diagnostic beginning in prefix storage cannot cross into output");
}

static void test_normal_arithmetic_environment()
{
    const int rounding_before = std::fegetround();
#if defined(__SSE__) || defined(_M_X64) || defined(_M_IX86_FP)
    const uint32_t mxcsr_before = _mm_getcsr();
    check((mxcsr_before & (UINT32_C(1) << 15)) == 0 &&
          (mxcsr_before & (UINT32_C(1) << 6)) == 0,
          "fast-math caller enters with FTZ and DAZ disabled");
#endif
    check(rounding_before == FE_TONEAREST,
          "test begins in round-to-nearest");
    check(g1_clearance_arithmetic_environment_is_supported(),
          "strict kernel accepts nearest and gradual underflow probes");
    check(std::fegetround() == rounding_before,
          "portable probe preserves caller rounding mode");
#if defined(__SSE__) || defined(_M_X64) || defined(_M_IX86_FP)
    check(_mm_getcsr() == mxcsr_before,
          "portable probe preserves caller MXCSR exactly");
#endif
    test_output_diagnostic_aliases_in_current_environment();
    test_limit_diagnostic_aliases_in_current_environment(true);
    test_history_diagnostic_aliases_in_current_environment();
    test_diagnostic_range_crossing_into_output();
}

static void test_rounding_mode_rejection(int requested_mode)
{
    const int original = std::fegetround();
    bool mode_set = false;
    bool statuses_correct = true;
    bool outputs_unchanged = true;
    bool mode_unchanged = true;
    {
        RoundingModeGuard guard;
        mode_set = std::fesetround(requested_mode) == 0 &&
                   std::fegetround() == requested_mode;
        if (mode_set) {
            const G1ClearanceBudget hostile = {
                UINT32_MAX, UINT32_MAX, UINT32_MAX,
                UINT32_MAX, UINT32_MAX, UINT32_MAX
            };
            for (int entry = 0; entry < PublicStatusEntryCount; ++entry) {
                PublicOutputs outputs = seeded_public_outputs();
                const ByteSnapshot<PublicOutputs> before(outputs);
                char error[64] = {};
                const G1ClearanceStatus status = invoke_public_status_entry(
                    static_cast<PublicStatusEntry>(entry),
                    outputs, hostile,
                    error, static_cast<int>(sizeof(error)));
                statuses_correct = statuses_correct &&
                    status == G1ClearanceArithmeticFailure;
                outputs_unchanged = outputs_unchanged &&
                    before.same(outputs);
                mode_unchanged = mode_unchanged &&
                    std::fegetround() == requested_mode;
            }
            test_output_diagnostic_aliases_in_current_environment();
            test_limit_diagnostic_aliases_in_current_environment(false);
            test_history_diagnostic_aliases_in_current_environment();
        }
    }
    check(std::fegetround() == original,
          "rounding guard restores the exact caller mode");
    check(mode_set, "requested non-nearest mode is available");
    check(statuses_correct,
          "every public status entry rejects non-nearest arithmetic");
    check(outputs_unchanged,
          "rounding rejection preserves every seeded output byte");
    check(mode_unchanged,
          "rejected calls never alter the inherited rounding mode");
}

#if defined(__SSE__) || defined(_M_X64) || defined(_M_IX86_FP)
static void test_mxcsr_rejection(uint32_t mask)
{
    const uint32_t original = _mm_getcsr();
    bool statuses_correct = true;
    bool outputs_unchanged = true;
    bool mode_unchanged = true;
    {
        MxcsrGuard guard;
        const uint32_t hostile_word = original | mask;
        _mm_setcsr(hostile_word);
        const G1ClearanceBudget hostile = {
            UINT32_MAX, UINT32_MAX, UINT32_MAX,
            UINT32_MAX, UINT32_MAX, UINT32_MAX
        };
        for (int entry = 0; entry < PublicStatusEntryCount; ++entry) {
            PublicOutputs outputs = seeded_public_outputs();
            const ByteSnapshot<PublicOutputs> before(outputs);
            char error[64] = {};
            const G1ClearanceStatus status = invoke_public_status_entry(
                static_cast<PublicStatusEntry>(entry),
                outputs, hostile,
                error, static_cast<int>(sizeof(error)));
            statuses_correct = statuses_correct &&
                status == G1ClearanceArithmeticFailure;
            outputs_unchanged = outputs_unchanged &&
                before.same(outputs);
            mode_unchanged = mode_unchanged &&
                _mm_getcsr() == hostile_word;
        }
        test_output_diagnostic_aliases_in_current_environment();
        test_limit_diagnostic_aliases_in_current_environment(false);
        test_history_diagnostic_aliases_in_current_environment();
    }
    check(_mm_getcsr() == original,
          "MXCSR guard restores the exact caller word");
    check(statuses_correct,
          "every public status entry rejects FTZ/DAZ arithmetic");
    check(outputs_unchanged,
          "FTZ/DAZ rejection preserves every seeded output byte");
    check(mode_unchanged,
          "rejected calls never alter inherited MXCSR");
}
#endif

static void test_arithmetic_environment_rejection_and_restoration()
{
    test_rounding_mode_rejection(FE_UPWARD);
    test_rounding_mode_rejection(FE_DOWNWARD);

#if defined(__SSE__) || defined(_M_X64) || defined(_M_IX86_FP)
    test_mxcsr_rejection(UINT32_C(1) << 15);
    test_mxcsr_rejection(UINT32_C(1) << 6);
    test_mxcsr_rejection(
        (UINT32_C(1) << 15) | (UINT32_C(1) << 6));
#endif

    check(g1_clearance_arithmetic_environment_is_supported(),
          "normal environment remains supported after hostile scopes");
    float output = float_from_bits(UINT32_C(0x41234567));
    char error[64] = {};
    check(g1_apply_swing_lift_y(
              output, float_from_bits(UINT32_C(0x3d0f5c28)),
              float_from_bits(UINT32_C(0x31000001)),
              error, static_cast<int>(sizeof(error))) == G1ClearanceOk &&
          float_bits(output) == UINT32_C(0x3d0f5c29),
          "valid call succeeds after every environment restoration");
}

int main(int argc, char** argv)
{
    if (argc == 2 && std::strcmp(argv[1], "--query-parity") == 0) {
        return run_query_parity_mode();
    }
    if (argc == 2 &&
        std::strcmp(argv[1], "--task34-sphere") == 0) {
        test_task34_smooth_plane_sphere_false_clear();
        return 0;
    }
    if (argc == 2 &&
        std::strcmp(argv[1], "--task34-capsule") == 0) {
        test_task34_finite_capsule_plane_and_reversal();
        return 0;
    }
    if (argc == 2 &&
        std::strcmp(argv[1], "--task34-domain-budget") == 0) {
        test_task34_footprint_domain_budget_and_transaction();
        return 0;
    }
    test_normal_arithmetic_environment();
    test_status_and_factory_contract();
    test_budget_contract();
    test_swing_lift_materializer();
    test_point_fixed_diagonal_and_determinism();
    test_point_special_exact_bit_regression();
    test_point_rejects_unproved_materialized_diagonal();
    test_point_rejects_false_rounded_diagonal_tie();
    test_point_domain_boundaries();
    test_point_input_and_field_rejection();
    test_point_exterior_height_independence();
    test_point_budget_preflight_and_aliases();
    test_point_float_output_guards();
    test_task34_smooth_plane_sphere_false_clear();
    test_task34_finite_capsule_plane_and_reversal();
    test_task34_footprint_domain_budget_and_transaction();
    test_arithmetic_environment_rejection_and_restoration();
    return 0;
}
