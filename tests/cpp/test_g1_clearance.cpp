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
            swing_family, limits,
            G1ClearanceInvalidField,
            "factory cap is admitted by its family");

        limits = factory;
        limits.*fields[index] = 0;
        require_budget_status(
            swing_family, limits,
            G1ClearanceInvalidField,
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
    check((float_bits(delta) & UINT32_C(7)) == 0,
          "smooth-plane delta has its low three mantissa bits cleared");
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
    check(static_cast<double>(center_x) ==
              4.0 * static_cast<double>(cell) &&
          static_cast<double>(center_z) ==
              4.0 * static_cast<double>(cell),
          "smooth-plane center uses the authoritative promoted node product");
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

    const double radial_spacing =
        0.5 * static_cast<double>(cell);
    const double promoted_radius = static_cast<double>(radius);
    const double promoted_center_y = static_cast<double>(center.y);
    const double promoted_surface = static_cast<double>(surface);
    const double radial_offsets[][2] = {
        {0.0, 0.0},
        {radial_spacing, 0.0},
        {-radial_spacing, 0.0},
        {0.0, radial_spacing},
        {0.0, -radial_spacing}
    };
    double rejected_lattice = std::numeric_limits<double>::infinity();
    for (size_t index = 0;
         index < sizeof(radial_offsets) / sizeof(radial_offsets[0]);
         ++index) {
        const double ox = radial_offsets[index][0];
        const double oz = radial_offsets[index][1];
        const double rho_square = ox * ox + oz * oz;
        const double body_y = promoted_center_y -
            std::sqrt(promoted_radius * promoted_radius - rho_square);
        const double terrain_y = promoted_surface + slope * ox;
        const double sampled = body_y - terrain_y;
        rejected_lattice = sampled < rejected_lattice
            ? sampled
            : rejected_lattice;
    }
    check(rejected_lattice > 0.0043 &&
              rejected_lattice < 0.0045,
          "rejected half-cell radial lattice falsely clears near +0.00436 m");

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
            const double intended =
                slope_x * px + slope_z * pz + intercept;
            const float stored = static_cast<float>(intended);
            check(static_cast<double>(stored) == intended,
                  "capsule plane fixture heights are exact binary32 dyadics");
            field.heights(x + z * field.nx) = stored;
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
    check(forward.witness.cell_x == 6 &&
              forward.witness.cell_z == 4 &&
              forward.witness.terrain_triangle_index == 1 &&
              forward.witness.patch_index == 1 &&
              forward.witness.candidate_kind == 0 &&
              forward.witness.candidate_subindex == 0 &&
              forward.work.subdivision_nodes == 0,
          "finite planar capsule selects the analytic B-cap face key");
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

    const vec3 axis_tangent_centers[] = {
        vec3(0.25f, 1.0f, 0.50f),
        vec3(0.75f, 1.0f, 0.50f),
        vec3(0.50f, 1.0f, 0.25f),
        vec3(0.50f, 1.0f, 0.75f)
    };
    for (size_t index = 0;
         index < sizeof(axis_tangent_centers) /
                     sizeof(axis_tangent_centers[0]);
         ++index) {
        G1ClearanceResult axis_output = seeded_result(
            152.0 + static_cast<double>(index));
        check(g1_sphere_clearance(
                  axis_output, g1_pose_clearance_budget(), field,
                  axis_tangent_centers[index], radius,
                  NULL, 0) == G1ClearanceOk,
              "sphere certifies each exact min/max axis tangency");
    }

    const vec3 capsule_endpoints[][2] = {
        {vec3(0.25f, 1.0f, 0.25f),
         vec3(0.25f, 1.0f, 0.75f)},
        {vec3(0.75f, 1.0f, 0.25f),
         vec3(0.75f, 1.0f, 0.75f)},
        {vec3(0.25f, 1.0f, 0.25f),
         vec3(0.75f, 1.0f, 0.25f)},
        {vec3(0.25f, 1.0f, 0.75f),
         vec3(0.75f, 1.0f, 0.75f)}
    };
    for (size_t index = 0;
         index < sizeof(capsule_endpoints) /
                     sizeof(capsule_endpoints[0]);
         ++index) {
        G1ClearanceResult forward = seeded_result(
            156.0 + static_cast<double>(index));
        G1ClearanceResult reverse = seeded_result(
            160.0 + static_cast<double>(index));
        check(g1_capsule_clearance(
                  forward, g1_pose_clearance_budget(), field,
                  capsule_endpoints[index][0],
                  capsule_endpoints[index][1], radius,
                  NULL, 0) == G1ClearanceOk &&
              g1_capsule_clearance(
                  reverse, g1_pose_clearance_budget(), field,
                  capsule_endpoints[index][1],
                  capsule_endpoints[index][0], radius,
                  NULL, 0) == G1ClearanceOk,
              "capsule certifies each exact min/max axis tangency");
        check(clearance_result_same(forward, reverse),
              "axis-tangent capsule reversal remains bit-identical");
    }

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

    const vec3 outward_centers[] = {
        vec3(std::nextafter(0.25f, 0.0f), 1.0f, 0.50f),
        vec3(std::nextafter(
                 0.75f, std::numeric_limits<float>::infinity()),
             1.0f, 0.50f),
        vec3(0.50f, 1.0f, std::nextafter(0.25f, 0.0f)),
        vec3(0.50f, 1.0f,
             std::nextafter(
                 0.75f, std::numeric_limits<float>::infinity()))
    };
    for (size_t index = 0;
         index < sizeof(outward_centers) /
                     sizeof(outward_centers[0]);
         ++index) {
        G1ClearanceResult rejected = seeded_result(
            165.0 + static_cast<double>(index));
        const ByteSnapshot<G1ClearanceResult> before(rejected);
        check(g1_sphere_clearance(
                  rejected, g1_pose_clearance_budget(), field,
                  outward_centers[index], radius,
                  NULL, 0) == G1ClearanceOutsideDomain,
              "one-ULP sphere excursion on each axis is outside-domain");
        check(before.same(rejected),
              "axis-specific outside-domain rejection is transactional");
    }

    const vec3 outward_capsules[][2] = {
        {
            vec3(std::nextafter(0.25f, 0.0f), 1.0f, 0.25f),
            vec3(std::nextafter(0.25f, 0.0f), 1.0f, 0.75f)
        },
        {
            vec3(std::nextafter(
                     0.75f, std::numeric_limits<float>::infinity()),
                 1.0f, 0.25f),
            vec3(std::nextafter(
                     0.75f, std::numeric_limits<float>::infinity()),
                 1.0f, 0.75f)
        },
        {
            vec3(0.25f, 1.0f, std::nextafter(0.25f, 0.0f)),
            vec3(0.75f, 1.0f, std::nextafter(0.25f, 0.0f))
        },
        {
            vec3(0.25f, 1.0f,
                 std::nextafter(
                     0.75f, std::numeric_limits<float>::infinity())),
            vec3(0.75f, 1.0f,
                 std::nextafter(
                     0.75f, std::numeric_limits<float>::infinity()))
        }
    };
    for (size_t index = 0;
         index < sizeof(outward_capsules) /
                     sizeof(outward_capsules[0]);
         ++index) {
        G1ClearanceResult forward = seeded_result(
            170.0 + static_cast<double>(index));
        G1ClearanceResult reverse = seeded_result(
            174.0 + static_cast<double>(index));
        const ByteSnapshot<G1ClearanceResult> forward_before(forward);
        const ByteSnapshot<G1ClearanceResult> reverse_before(reverse);
        check(g1_capsule_clearance(
                  forward, g1_pose_clearance_budget(), field,
                  outward_capsules[index][0],
                  outward_capsules[index][1], radius,
                  NULL, 0) == G1ClearanceOutsideDomain &&
              g1_capsule_clearance(
                  reverse, g1_pose_clearance_budget(), field,
                  outward_capsules[index][1],
                  outward_capsules[index][0], radius,
                  NULL, 0) == G1ClearanceOutsideDomain,
              "one-ULP capsule excursion on each axis is outside-domain");
        check(forward_before.same(forward) &&
                  reverse_before.same(reverse),
              "outside capsule rejection is transactional under reversal");
    }

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

static void task34_require_capsule_oracle(
    const heightfield& field,
    vec3 endpoint_a,
    vec3 endpoint_b,
    float radius,
    double oracle,
    const char* status_message,
    const char* oracle_message)
{
    G1ClearanceResult output = seeded_result(167.0);
    char error[256] = {};
    const G1ClearanceStatus status = g1_capsule_clearance(
        output, g1_pose_clearance_budget(), field,
        endpoint_a, endpoint_b, radius,
        error, static_cast<int>(sizeof(error)));
    check(status == G1ClearanceOk, status_message);
    check(output.lower_bound_m <= oracle &&
              oracle <= output.witness_upper_m &&
              output.witness_upper_m - output.lower_bound_m <=
                  G1ClearanceMaximumCertificateWidthM,
          oracle_message);
}

enum Task34ExactEdgeClamp
{
    Task34ExactEdgeLower,
    Task34ExactEdgeStationary,
    Task34ExactEdgeUpper
};

static Task34ExactEdgeClamp task34_classify_exact_edge_clamp(
    long double first_x,
    long double first_y,
    long double first_z,
    long double second_x,
    long double second_y,
    long double second_z,
    long double radius)
{
    const long double dx = second_x - first_x;
    const long double dz = second_z - first_z;
    const long double length = std::sqrt(dx * dx + dz * dz);
    check(length > 0.0L,
          "exact edge-clamp classifier requires projected length");
    const long double direction_x = dx / length;
    const long double direction_z = dz / length;
    const long double s0 =
        first_x * direction_x + first_z * direction_z;
    const long double s1 = s0 + length;
    const long double cross = first_x * dz - first_z * dx;
    const long double q_square =
        (cross * cross) / (length * length);
    check(q_square <= radius * radius,
          "exact edge-clamp fixture intersects the projected disk");
    const long double line_radius =
        std::sqrt(radius * radius - q_square);
    const long double slope = (second_y - first_y) / length;
    const long double stationary =
        -slope * line_radius / std::sqrt(1.0L + slope * slope);
    const long double lower = s0 > -line_radius ? s0 : -line_radius;
    const long double upper = s1 < line_radius ? s1 : line_radius;
    check(lower <= upper,
          "exact edge-clamp fixture has a nonempty clipped interval");
    if (stationary < lower) {
        return Task34ExactEdgeLower;
    }
    if (stationary > upper) {
        return Task34ExactEdgeUpper;
    }
    return Task34ExactEdgeStationary;
}

static bool task34_long_double_near(
    long double left,
    long double right,
    long double tolerance)
{
    return std::fabs(left - right) <= tolerance;
}

static void task34_require_public_witness_enclosure(
    const heightfield& field,
    vec3 canonical_endpoint_0,
    vec3 canonical_endpoint_1,
    float radius,
    const G1ClearanceResult& result)
{
    const G1ClearanceWitness& witness = result.witness;
    check(witness.cell_x >= 0 && witness.cell_x + 1 < field.nx &&
              witness.cell_z >= 0 && witness.cell_z + 1 < field.nz &&
              witness.terrain_triangle_index <= 1,
          "public witness key selects a valid fixed-diagonal triangle");

    const int x0_index = witness.cell_x;
    const int x1_index = witness.cell_x + 1;
    const int z0_index = witness.cell_z;
    const int z1_index = witness.cell_z + 1;
    const long double origin_x =
        static_cast<long double>(field.origin_x);
    const long double origin_z =
        static_cast<long double>(field.origin_z);
    const long double cell =
        static_cast<long double>(field.cell_size);
    const long double x0 = origin_x +
        static_cast<long double>(x0_index) * cell;
    const long double x1 = origin_x +
        static_cast<long double>(x1_index) * cell;
    const long double z0 = origin_z +
        static_cast<long double>(z0_index) * cell;
    const long double z1 = origin_z +
        static_cast<long double>(z1_index) * cell;
    const long double h00 = static_cast<long double>(
        field.heights(x0_index + z0_index * field.nx));
    const long double h10 = static_cast<long double>(
        field.heights(x1_index + z0_index * field.nx));
    const long double h01 = static_cast<long double>(
        field.heights(x0_index + z1_index * field.nx));
    const long double h11 = static_cast<long double>(
        field.heights(x1_index + z1_index * field.nx));

    long double triangle_x[3] = {};
    long double triangle_y[3] = {};
    long double triangle_z[3] = {};
    if (witness.terrain_triangle_index == 0) {
        triangle_x[0] = x0;
        triangle_y[0] = h00;
        triangle_z[0] = z0;
        triangle_x[1] = x1;
        triangle_y[1] = h10;
        triangle_z[1] = z0;
        triangle_x[2] = x1;
        triangle_y[2] = h11;
        triangle_z[2] = z1;
    } else {
        triangle_x[0] = x0;
        triangle_y[0] = h00;
        triangle_z[0] = z0;
        triangle_x[1] = x1;
        triangle_y[1] = h11;
        triangle_z[1] = z1;
        triangle_x[2] = x0;
        triangle_y[2] = h01;
        triangle_z[2] = z1;
    }

    const long double weights[3] = {
        static_cast<long double>(witness.terrain_weight_0),
        static_cast<long double>(witness.terrain_weight_1),
        static_cast<long double>(witness.terrain_weight_2)
    };
    const long double t =
        static_cast<long double>(witness.segment_parameter);
    check(t >= 0.0L && t <= 1.0L &&
              weights[0] >= 0.0L &&
              weights[1] >= 0.0L &&
              weights[2] >= 0.0L,
          "public solver witness diagnostics are parameter-feasible");
    const long double sum = weights[0] + weights[1] + weights[2];
    check(task34_long_double_near(sum, 1.0L, 0x1p-48L),
          "public solver witness weights reconstruct a unit barycentric sum");

    long double surface_x = 0.0L;
    long double surface_y = 0.0L;
    long double surface_z = 0.0L;
    for (int index = 0; index < 3; ++index) {
        surface_x += weights[index] * triangle_x[index];
        surface_y += weights[index] * triangle_y[index];
        surface_z += weights[index] * triangle_z[index];
    }
    const long double one_minus_t = 1.0L - t;
    const long double center_x =
        one_minus_t * static_cast<long double>(canonical_endpoint_0.x) +
        t * static_cast<long double>(canonical_endpoint_1.x);
    const long double center_y =
        one_minus_t * static_cast<long double>(canonical_endpoint_0.y) +
        t * static_cast<long double>(canonical_endpoint_1.y);
    const long double center_z =
        one_minus_t * static_cast<long double>(canonical_endpoint_0.z) +
        t * static_cast<long double>(canonical_endpoint_1.z);
    const long double coordinate_tolerance = 0x1p-46L;
    check(task34_long_double_near(
              static_cast<long double>(witness.surface_x),
              surface_x, coordinate_tolerance) &&
          task34_long_double_near(
              static_cast<long double>(witness.surface_y),
              surface_y, coordinate_tolerance) &&
          task34_long_double_near(
              static_cast<long double>(witness.surface_z),
              surface_z, coordinate_tolerance),
          "public witness surface reconstructs from its fixed triangle and weights");
    check(double_bits(witness.body_x) ==
              double_bits(witness.surface_x) &&
          double_bits(witness.body_z) ==
              double_bits(witness.surface_z),
          "public witness body and surface share identical XZ bits");

    const long double dx = surface_x - center_x;
    const long double dz = surface_z - center_z;
    const long double dy =
        static_cast<long double>(witness.body_y) - center_y;
    const long double radius_ld = static_cast<long double>(radius);
    const long double radius_square = radius_ld * radius_ld;
    const long double rho_square = dx * dx + dz * dz;
    const long double feasibility_tolerance = 0x1p-46L;
    check(rho_square <= radius_square + feasibility_tolerance &&
              rho_square + dy * dy <=
                  radius_square + feasibility_tolerance,
          "independent reconstruction places the witness inside the capsule ball");
    const long double exact_vertical = std::sqrt(
        radius_square - (rho_square < radius_square
            ? rho_square
            : radius_square));
    const long double reconstructed_objective =
        center_y - surface_y - exact_vertical;
    const long double diagnostic_objective =
        static_cast<long double>(witness.body_y) - surface_y;
    const long double lower =
        static_cast<long double>(result.lower_bound_m);
    const long double upper =
        static_cast<long double>(result.witness_upper_m);
    check(lower <= reconstructed_objective + feasibility_tolerance &&
              reconstructed_objective <= upper + feasibility_tolerance &&
              diagnostic_objective <= upper + feasibility_tolerance,
          "independent witness objective lies inside the public certificate");
    check(upper - diagnostic_objective <= 0x1p-38L,
          "public witness upper tightly outward-encloses its diagnostic objective");
}

static void test_task34_fixed_diagonal_and_rank_cases()
{
    heightfield diagonal;
    point_make_field(diagonal);
    diagonal.heights(0) = 0.0f;
    diagonal.heights(1) = 2.0f;
    diagonal.heights(2) = 4.0f;
    diagonal.heights(3) = 10.0f;
    const vec3 diagonal_center(0.5f, 20.0f, 0.5f);
    const float diagonal_radius = 0.05f;
    const vec3 triangle_centers[] = {
        vec3(0.75f, 20.0f, 0.25f),
        vec3(0.25f, 20.0f, 0.75f)
    };
    const double triangle_oracles[] = {
        20.0 - 3.5 -
            static_cast<double>(diagonal_radius) * std::sqrt(69.0),
        20.0 - 4.5 -
            static_cast<double>(diagonal_radius) * std::sqrt(53.0)
    };
    for (size_t index = 0;
         index < sizeof(triangle_centers) /
                     sizeof(triangle_centers[0]);
         ++index) {
        G1ClearanceResult triangle_output = seeded_result(
            169.0 + static_cast<double>(index));
        check(g1_sphere_clearance(
                  triangle_output, g1_pose_clearance_budget(), diagonal,
                  triangle_centers[index], diagonal_radius,
                  NULL, 0) == G1ClearanceOk,
              "strict fixed-diagonal triangle-center sphere certifies");
        check(triangle_output.lower_bound_m <=
                  triangle_oracles[index] &&
              triangle_oracles[index] <=
                  triangle_output.witness_upper_m &&
              triangle_output.witness.candidate_kind == 0 &&
              triangle_output.work.subdivision_nodes == 0,
              "strict T0/T1 sphere encloses its plane oracle");
    }
    const double diagonal_oracle =
        static_cast<double>(diagonal_center.y) - 5.0 -
        static_cast<double>(diagonal_radius) * std::sqrt(51.0);
    G1ClearanceResult diagonal_output = seeded_result(173.0);
    check(g1_sphere_clearance(
              diagonal_output, g1_pose_clearance_budget(), diagonal,
              diagonal_center, diagonal_radius, NULL, 0) ==
              G1ClearanceOk,
          "fixed-diagonal constrained-edge sphere must certify");
    check(diagonal_output.lower_bound_m <= diagonal_oracle &&
              diagonal_oracle <= diagonal_output.witness_upper_m &&
              diagonal_output.witness_upper_m -
                      diagonal_output.lower_bound_m <=
                  G1ClearanceMaximumCertificateWidthM,
          "fixed-diagonal constrained-edge certificate encloses r*sqrt(51)");
    check(diagonal_output.witness.primitive_index == 0 &&
              diagonal_output.witness.cell_x == 0 &&
              diagonal_output.witness.cell_z == 0 &&
              diagonal_output.witness.terrain_triangle_index == 0 &&
              diagonal_output.witness.patch_index == 7 &&
              diagonal_output.witness.candidate_kind == 1 &&
              diagonal_output.witness.candidate_subindex == 3,
          "fixed-diagonal fixture pins the public stationary-edge key");

    heightfield clamp;
    point_make_field(clamp, 3, 3, 0.0f, 0.0f, 1.0f);
    clamp.heights(1 + clamp.nx) = 1.0f;
    G1ClearanceResult clamp_output = seeded_result(177.0);
    check(g1_sphere_clearance(
              clamp_output, g1_pose_clearance_budget(), clamp,
              vec3(1.0f, 2.0f, 1.0f), 0.25f,
              NULL, 0) == G1ClearanceOk,
          "edge stationary/below/above clamp fixture must certify");
    const G1ClearanceWork expected_clamp_work = {0, 4, 8, 64, 256, 0};
    check(clamp_output.lower_bound_m <= 0.75 &&
              0.75 <= clamp_output.witness_upper_m &&
              double_bits(clamp_output.lower_bound_m) ==
                  UINT64_C(0x3fe7ffffbfffffc5) &&
              double_bits(clamp_output.witness_upper_m) ==
                  UINT64_C(0x3fe800000000000b) &&
              clamp_output.witness.cell_x == 0 &&
              clamp_output.witness.cell_z == 0 &&
              clamp_output.witness.terrain_triangle_index == 0 &&
              clamp_output.witness.patch_index == 0 &&
              clamp_output.witness.candidate_kind == 1 &&
              clamp_output.witness.candidate_subindex == 5 &&
              clearance_work_same(
                  clamp_output.work, expected_clamp_work),
          "edge clamp fixture selects the stable upper-endpoint edge key");
    check(task34_classify_exact_edge_clamp(
              0.0L, 2.0L, 1.0L,
              0.0L, 1.0L, 0.0L,
              0.25L) == Task34ExactEdgeUpper,
          "apex edge 1 independently exercises the upper clamp outcome");
    check(task34_classify_exact_edge_clamp(
              0.0L, 1.0L, 0.0L,
              1.0L, 2.0L, 1.0L,
              0.25L) == Task34ExactEdgeLower,
          "apex edge 2 independently exercises the lower clamp outcome");
    // Both edge outcomes meet at T0 vertex 2.  The public key must name the
    // earlier upper endpoint (subindex 5), not the tied lower endpoint
    // (subindex 7); a public local-1 winner is therefore neither expected nor
    // manufactured by weakening the stable key order.

    heightfield flat;
    point_make_field(flat, 9, 9, 0.0f, 0.0f, 0.25f);
    const float radius = 0.125f;
    const vec3 point(1.0f, 0.75f, 1.0f);
    task34_require_capsule_oracle(
        flat, point, point, radius,
        static_cast<double>(point.y) - static_cast<double>(radius),
        "A==B capsule must certify through rank degeneracy",
        "A==B capsule encloses the flat-plane sphere oracle");

    G1ClearanceResult sphere = seeded_result(179.0);
    G1ClearanceResult zero_capsule = seeded_result(181.0);
    check(g1_sphere_clearance(
              sphere, g1_pose_clearance_budget(), flat,
              point, radius, NULL, 0) == G1ClearanceOk &&
          g1_capsule_clearance(
              zero_capsule, g1_pose_clearance_budget(), flat,
              point, point, radius, NULL, 0) == G1ClearanceOk,
          "sphere and zero-length capsule both certify");
    check(clearance_result_same(sphere, zero_capsule),
          "sphere and A==B capsule are bit-identical");

    const vec3 parallel_a(0.5f, 0.5f, 0.75f);
    const vec3 parallel_b(1.5f, 0.5f, 1.25f);
    task34_require_capsule_oracle(
        flat, parallel_a, parallel_b, radius,
        0.5 - static_cast<double>(radius),
        "terrain-parallel capsule must certify",
        "terrain-parallel capsule encloses its flat-plane oracle");

    const vec3 coplanar_a(0.5f, 0.0f, 0.75f);
    const vec3 coplanar_b(1.5f, 0.0f, 1.25f);
    task34_require_capsule_oracle(
        flat, coplanar_a, coplanar_b, radius,
        -static_cast<double>(radius),
        "terrain-coplanar capsule must certify",
        "terrain-coplanar capsule encloses its negative-radius oracle");

    const vec3 vertical_a(1.0f, 0.25f, 1.0f);
    const vec3 vertical_b(1.0f, 0.75f, 1.0f);
    task34_require_capsule_oracle(
        flat, vertical_a, vertical_b, radius,
        static_cast<double>(vertical_a.y) -
            static_cast<double>(radius),
        "vertical-projection capsule must certify",
        "vertical projected edges retain the nonzero sphere term");

    // The disk is tangent to multiple internal grid edges/vertices.  Those
    // pairs must remain feasible even though the flat global witness is at
    // the center.
    const vec3 tangent_center(1.0f, 0.75f, 1.0f);
    const float tangent_radius = 0.25f;
    G1ClearanceResult tangent = seeded_result(191.0);
    check(g1_sphere_clearance(
              tangent, g1_pose_clearance_budget(), flat,
              tangent_center, tangent_radius, NULL, 0) == G1ClearanceOk,
          "exact projected disk tangency remains feasible");
    const double tangent_oracle =
        static_cast<double>(tangent_center.y) -
        static_cast<double>(tangent_radius);
    check(tangent.lower_bound_m <= tangent_oracle &&
              tangent_oracle <= tangent.witness_upper_m,
          "exact disk tangency does not discard the flat-plane minimum");
}

static void test_task34_membership_uncertainty_and_witness_enclosure()
{
    heightfield field;
    point_make_field(field, 3, 3, 0.0f, 0.0f, 1.0f);
    const vec3 center(0.5f, 2.0f, 0.5f);
    const float radius = 0.25f;

    // On flat T0 the stationary terrain point is (0.5,0.5), with exact
    // barycentrics (0.5,0,0.5).  The zero weight lies on the fixed diagonal,
    // so membership must retain equality conservatively instead of mapping
    // the face to outside.
    G1ClearanceBudget zero_subdivision = g1_pose_clearance_budget();
    zero_subdivision.maximum_subdivision_nodes = 0;
    G1ClearanceResult bounded = seeded_result(192.0);
    check(g1_sphere_clearance(
              bounded, zero_subdivision, field,
              center, radius, NULL, 0) == G1ClearanceOk,
          "membership-straddling face closes conservatively without child work");
    check(double_bits(bounded.lower_bound_m) ==
              UINT64_C(0x3ffbfffffffffff5) &&
          double_bits(bounded.witness_upper_m) ==
              UINT64_C(0x3ffc00000000000b) &&
          bounded.witness.primitive_index == 0 &&
          bounded.witness.cell_x == 0 &&
          bounded.witness.cell_z == 0 &&
          bounded.witness.terrain_triangle_index == 0 &&
          bounded.witness.patch_index == 0 &&
          bounded.witness.candidate_kind == 0 &&
          bounded.witness.candidate_subindex == 0,
          "membership-straddling face pins its conservative certificate and key");
    const G1ClearanceWork expected_work = {0, 1, 2, 16, 64, 0};
    check(clearance_work_same(bounded.work, expected_work) &&
              double_bits(bounded.witness.segment_parameter) ==
                  UINT64_C(0x0000000000000000) &&
              double_bits(bounded.witness.terrain_weight_0) ==
                  UINT64_C(0x3fe0000000000000) &&
              double_bits(bounded.witness.terrain_weight_1) ==
                  UINT64_C(0x0000000000000000) &&
              double_bits(bounded.witness.terrain_weight_2) ==
                  UINT64_C(0x3fe0000000000000),
          "membership-straddling face exposes exact public feasibility diagnostics");
    const long double exact_oracle = 1.75L;
    check(static_cast<long double>(bounded.lower_bound_m) <= exact_oracle &&
              exact_oracle <=
                  static_cast<long double>(bounded.witness_upper_m),
          "membership-straddling face encloses its independent flat oracle");
    task34_require_public_witness_enclosure(
        field, center, center, radius, bounded);

    G1ClearanceResult default_budget = seeded_result(193.0);
    check(g1_sphere_clearance(
              default_budget, g1_pose_clearance_budget(), field,
              center, radius, NULL, 0) == G1ClearanceOk &&
          clearance_result_same(bounded, default_budget),
          "membership conservative mapping is bit-identical across node caps");

    // Raising only Y preserves this exact projected membership boundary while
    // widening its absolute binary64 enclosure.  With no subdivision nodes,
    // the conservative uncertain-face mapping must reach the bounded fallback
    // and fail transactionally there.  Incorrectly treating uncertainty as
    // inside instead reaches the distinct analytic-width failure path.
    G1ClearanceResult routed = seeded_result(196.0);
    const ByteSnapshot<G1ClearanceResult> routed_before(routed);
    char routed_error[128] = {};
    check(g1_sphere_clearance(
              routed, zero_subdivision, field,
              vec3(0.5f,
                   float_from_bits(UINT32_C(0x4e000000)),
                   0.5f),
              radius, routed_error,
              static_cast<int>(sizeof(routed_error))) ==
              G1ClearanceUncertified,
          "membership uncertainty with a wide enclosure fails closed");
    check(routed_before.same(routed) &&
              std::strcmp(
                  routed_error,
                  "G1 capsule subdivision cap exhausted") == 0,
          "membership uncertainty publicly routes through bounded fallback");
}

static void test_task34_exact_zero_objective_tie_characterization()
{
    heightfield field;
    point_make_field(field, 5, 5, 0.0f, 0.0f, 0.25f);
    const vec3 endpoint_a(0.5f, 0.125f, 0.5f);
    const vec3 endpoint_b(0.75f, 0.125f, 0.5f);
    const float radius = 0.125f;
    G1ClearanceResult forward = seeded_result(194.0);
    G1ClearanceResult reverse = seeded_result(195.0);
    check(g1_capsule_clearance(
              forward, g1_pose_clearance_budget(), field,
              endpoint_a, endpoint_b, radius, NULL, 0) ==
              G1ClearanceOk &&
          g1_capsule_clearance(
              reverse, g1_pose_clearance_budget(), field,
              endpoint_b, endpoint_a, radius, NULL, 0) ==
              G1ClearanceOk &&
          clearance_result_same(forward, reverse),
          "exact-zero capsule objective remains bit-identical under reversal");
    check(double_bits(forward.lower_bound_m) ==
              UINT64_C(0xbcbe000000000002) &&
          double_bits(forward.witness_upper_m) ==
              UINT64_C(0x3ca8000000000001) &&
          forward.witness.cell_x == 1 &&
          forward.witness.cell_z == 1 &&
          forward.witness.terrain_triangle_index == 0 &&
          forward.witness.patch_index == 0 &&
          forward.witness.candidate_kind == 0 &&
          forward.witness.candidate_subindex == 0,
          "exact-zero objective pins its stable outward value and complete key");
    check(forward.witness_upper_m > 0.0 &&
              !std::signbit(forward.witness_upper_m),
          "reachable exact-zero objective expands to positive nonzero binary64");
    // This is the focused public signed-zero characterization: outward
    // witness construction makes the exact-zero objective nonzero before
    // aggregation, so +0/-0 objective bits are not reached by this canonical
    // tie fixture.  Tie comparisons still must implement the design's
    // identical-bit rule rather than relying on numeric equality.
}

static void task34_set_coordinate(
    vec3& value,
    int coordinate,
    float replacement)
{
    if (coordinate == 0) {
        value.x = replacement;
    } else if (coordinate == 1) {
        value.y = replacement;
    } else {
        value.z = replacement;
    }
}

static void task34_require_invalid_field_primitives(
    const heightfield& field,
    const char* message)
{
    const vec3 endpoint_a(0.375f, 1.0f, 0.375f);
    const vec3 endpoint_b(0.625f, 1.25f, 0.625f);
    G1ClearanceResult sphere = seeded_result(192.0);
    G1ClearanceResult capsule = seeded_result(193.0);
    const ByteSnapshot<G1ClearanceResult> sphere_before(sphere);
    const ByteSnapshot<G1ClearanceResult> capsule_before(capsule);
    check(g1_sphere_clearance(
              sphere, g1_pose_clearance_budget(), field,
              endpoint_a, 0.125f, NULL, 0) ==
              G1ClearanceInvalidField &&
          g1_capsule_clearance(
              capsule, g1_pose_clearance_budget(), field,
              endpoint_a, endpoint_b, 0.125f,
              NULL, 0) == G1ClearanceInvalidField,
          message);
    check(sphere_before.same(sphere) &&
              capsule_before.same(capsule),
          "invalid-field sphere/capsule failures are transactional");
}

static void test_task34_input_field_and_exterior_matrix()
{
    heightfield valid;
    point_make_field(valid, 5, 5, 0.0f, 0.0f, 0.25f);
    const vec3 endpoint_a(0.375f, 1.0f, 0.375f);
    const vec3 endpoint_b(0.625f, 1.25f, 0.625f);
    const float invalid_values[] = {
        float_from_bits(UINT32_C(0x00000000)),
        float_from_bits(UINT32_C(0x80000000)),
        float_from_bits(UINT32_C(0xbe000000)),
        float_from_bits(UINT32_C(0x00000001)),
        float_from_bits(UINT32_C(0x80000001)),
        float_from_bits(UINT32_C(0x7f800000)),
        float_from_bits(UINT32_C(0xff800000)),
        float_from_bits(UINT32_C(0x7fc00001))
    };
    for (size_t value_index = 0;
         value_index < sizeof(invalid_values) /
                           sizeof(invalid_values[0]);
         ++value_index) {
        G1ClearanceResult sphere_radius = seeded_result(
            194.0 + static_cast<double>(value_index));
        G1ClearanceResult capsule_radius = seeded_result(
            204.0 + static_cast<double>(value_index));
        const ByteSnapshot<G1ClearanceResult> sphere_radius_before(
            sphere_radius);
        const ByteSnapshot<G1ClearanceResult> capsule_radius_before(
            capsule_radius);
        check(g1_sphere_clearance(
                  sphere_radius, g1_pose_clearance_budget(), valid,
                  endpoint_a, invalid_values[value_index],
                  NULL, 0) == G1ClearanceInvalidInput &&
              g1_capsule_clearance(
                  capsule_radius, g1_pose_clearance_budget(), valid,
                  endpoint_a, endpoint_b,
                  invalid_values[value_index],
                  NULL, 0) == G1ClearanceInvalidInput,
              "sphere/capsule reject zero, negative, subnormal, and nonfinite radii");
        check(sphere_radius_before.same(sphere_radius) &&
                  capsule_radius_before.same(capsule_radius),
              "invalid radius failures preserve sphere/capsule outputs");

        if (value_index < 3) {
            continue;
        }
        for (int coordinate = 0; coordinate < 3; ++coordinate) {
            vec3 invalid_endpoint = endpoint_a;
            task34_set_coordinate(
                invalid_endpoint, coordinate,
                invalid_values[value_index]);
            G1ClearanceResult sphere = seeded_result(
                214.0 + static_cast<double>(coordinate));
            G1ClearanceResult forward = seeded_result(
                218.0 + static_cast<double>(coordinate));
            G1ClearanceResult reverse = seeded_result(
                222.0 + static_cast<double>(coordinate));
            const ByteSnapshot<G1ClearanceResult> sphere_before(sphere);
            const ByteSnapshot<G1ClearanceResult> forward_before(forward);
            const ByteSnapshot<G1ClearanceResult> reverse_before(reverse);
            check(g1_sphere_clearance(
                      sphere, g1_pose_clearance_budget(), valid,
                      invalid_endpoint, 0.125f,
                      NULL, 0) == G1ClearanceInvalidInput &&
                  g1_capsule_clearance(
                      forward, g1_pose_clearance_budget(), valid,
                      invalid_endpoint, endpoint_b, 0.125f,
                      NULL, 0) == G1ClearanceInvalidInput &&
                  g1_capsule_clearance(
                      reverse, g1_pose_clearance_budget(), valid,
                      endpoint_b, invalid_endpoint, 0.125f,
                      NULL, 0) == G1ClearanceInvalidInput,
                  "sphere/capsule reject subnormal and nonfinite XYZ");
            check(sphere_before.same(sphere) &&
                      forward_before.same(forward) &&
                      reverse_before.same(reverse),
                  "invalid XYZ failures preserve all primitive outputs");
        }
    }

    heightfield version_one = valid;
    version_one.version = 1;
    task34_require_invalid_field_primitives(
        version_one, "sphere/capsule reject legacy v1 fields");

    heightfield wrong_shape = valid;
    --wrong_shape.heights.size;
    task34_require_invalid_field_primitives(
        wrong_shape, "sphere/capsule reject wrong-sized height storage");

    heightfield null_storage = valid;
    float* const saved_storage = null_storage.heights.data;
    null_storage.heights.data = NULL;
    task34_require_invalid_field_primitives(
        null_storage, "sphere/capsule reject null height storage");
    null_storage.heights.data = saved_storage;

    heightfield exterior_low = valid;
    heightfield exterior_high = valid;
    exterior_low.exterior_height = -10.0f;
    exterior_high.exterior_height = 10.0f;
    for (int index = 0; index < exterior_low.heights.size; ++index) {
        const float height = 0.03125f *
            static_cast<float>(index % exterior_low.nx);
        exterior_low.heights(index) = height;
        exterior_high.heights(index) = height;
    }
    G1ClearanceResult low_sphere = seeded_result(228.0);
    G1ClearanceResult high_sphere = seeded_result(229.0);
    G1ClearanceResult low_capsule = seeded_result(230.0);
    G1ClearanceResult high_capsule = seeded_result(231.0);
    const vec3 exterior_endpoint_b(
        endpoint_a.x, endpoint_b.y, endpoint_a.z);
    const G1ClearanceStatus low_sphere_status = g1_sphere_clearance(
        low_sphere, g1_pose_clearance_budget(), exterior_low,
        endpoint_a, 0.125f, NULL, 0);
    const G1ClearanceStatus high_sphere_status = g1_sphere_clearance(
        high_sphere, g1_pose_clearance_budget(), exterior_high,
        endpoint_a, 0.125f, NULL, 0);
    const G1ClearanceStatus low_capsule_status = g1_capsule_clearance(
        low_capsule, g1_pose_clearance_budget(), exterior_low,
        endpoint_a, exterior_endpoint_b, 0.125f, NULL, 0);
    const G1ClearanceStatus high_capsule_status = g1_capsule_clearance(
        high_capsule, g1_pose_clearance_budget(), exterior_high,
        exterior_endpoint_b, endpoint_a, 0.125f, NULL, 0);
    check(low_sphere_status == G1ClearanceOk &&
              high_sphere_status == G1ClearanceOk &&
              low_capsule_status == G1ClearanceOk &&
              high_capsule_status == G1ClearanceOk,
          "in-domain primitives certify for both exterior sentinels");
    check(clearance_result_same(low_sphere, high_sphere) &&
              clearance_result_same(low_capsule, high_capsule),
          "sphere/capsule output ignores exterior height and reversal");

    const vec3 outside_a(0.0625f, 1.0f, 0.375f);
    const vec3 outside_b(0.1875f, 1.25f, 0.625f);
    G1ClearanceResult low_outside = seeded_result(232.0);
    G1ClearanceResult high_outside = seeded_result(233.0);
    const ByteSnapshot<G1ClearanceResult> low_outside_before(low_outside);
    const ByteSnapshot<G1ClearanceResult> high_outside_before(high_outside);
    check(g1_capsule_clearance(
              low_outside, g1_pose_clearance_budget(), exterior_low,
              outside_a, outside_b, 0.125f,
              NULL, 0) == G1ClearanceOutsideDomain &&
          g1_capsule_clearance(
              high_outside, g1_pose_clearance_budget(), exterior_high,
              outside_b, outside_a, 0.125f,
              NULL, 0) == G1ClearanceOutsideDomain,
          "outside capsule status ignores exterior height and reversal");
    check(low_outside_before.same(low_outside) &&
              high_outside_before.same(high_outside),
          "outside capsule remains transactional for both exteriors");

    const float minimum_normal =
        float_from_bits(UINT32_C(0x00800000));
    heightfield minimum_cell;
    point_make_field(
        minimum_cell, 2, 2,
        0.0f, 0.0f, minimum_normal);
    G1ClearanceResult minimum_cell_output = seeded_result(234.0);
    const ByteSnapshot<G1ClearanceResult> minimum_cell_before(
        minimum_cell_output);
    check(g1_sphere_clearance(
              minimum_cell_output, g1_pose_clearance_budget(),
              minimum_cell,
              vec3(minimum_normal, 1.0f, minimum_normal),
              minimum_normal, NULL, 0) ==
              G1ClearanceOutsideDomain,
          "minimum-positive-normal cell rejects an oversized footprint promptly");
    check(minimum_cell_before.same(minimum_cell_output),
          "minimum-cell outside rejection preserves primitive output");

    const float maximum_float =
        float_from_bits(UINT32_C(0x7f7fffff));
    G1ClearanceResult maximum_radius = seeded_result(235.0);
    G1ClearanceResult maximum_endpoint = seeded_result(236.0);
    const ByteSnapshot<G1ClearanceResult> maximum_radius_before(
        maximum_radius);
    const ByteSnapshot<G1ClearanceResult> maximum_endpoint_before(
        maximum_endpoint);
    check(g1_sphere_clearance(
              maximum_radius, g1_pose_clearance_budget(), valid,
              endpoint_a, maximum_float, NULL, 0) ==
              G1ClearanceOutsideDomain &&
          g1_capsule_clearance(
              maximum_endpoint, g1_pose_clearance_budget(), valid,
              vec3(maximum_float, 1.0f, maximum_float),
              endpoint_b, 0.125f, NULL, 0) ==
              G1ClearanceOutsideDomain,
          "FLT_MAX radius and endpoints reject outside-domain promptly");
    check(maximum_radius_before.same(maximum_radius) &&
              maximum_endpoint_before.same(maximum_endpoint),
          "FLT_MAX outside rejections preserve primitive outputs");

    heightfield wide_axis;
    point_make_field(
        wide_axis, 3, 3,
        -maximum_float, -maximum_float, maximum_float);
    G1ClearanceResult wide_output = seeded_result(237.0);
    check(g1_capsule_clearance(
              wide_output, g1_pose_clearance_budget(), wide_axis,
              vec3(-0.125f, 1.0f, -0.125f),
              vec3(0.125f, 1.25f, 0.125f),
              0.125f, NULL, 0) == G1ClearanceOk,
          "wide-axis endpoint subtraction is certified without binary32 overflow");
    check(wide_output.lower_bound_m <= 0.875 &&
              0.875 <= wide_output.witness_upper_m,
          "wide-axis capsule encloses its flat-plane endpoint oracle");
}

static void test_task34_domain_and_malformed_height_matrix()
{
    heightfield nonrepresentable;
    const float cell = float_from_bits(UINT32_C(0x3e4ccccd));
    point_make_field(nonrepresentable, 4, 4, 0.0f, 0.0f, cell);
    const volatile double maximum_x =
        3.0 * static_cast<double>(cell);
    check(maximum_x !=
              static_cast<double>(static_cast<float>(maximum_x)),
          "sphere maximum-tangency fixture has a non-binary32 maximum");
    const float radius = cell;
    const vec3 tangent(
        2.0f * cell, 1.0f, 2.0f * cell);
    check(static_cast<double>(tangent.x) ==
              2.0 * static_cast<double>(cell) &&
          static_cast<double>(tangent.x) +
                  static_cast<double>(radius) == maximum_x &&
          static_cast<double>(tangent.x) -
                  static_cast<double>(radius) ==
              static_cast<double>(cell),
          "non-binary32 maximum tangency is exact in promoted operands");
    G1ClearanceResult tangent_output = seeded_result(193.0);
    check(g1_sphere_clearance(
              tangent_output, g1_pose_clearance_budget(),
              nonrepresentable, tangent, radius,
              NULL, 0) == G1ClearanceOk,
          "exact expansion tangency to non-binary32 maximum certifies");

    const vec3 outside(
        std::nextafter(tangent.x,
                       std::numeric_limits<float>::infinity()),
        tangent.y, tangent.z);
    G1ClearanceResult outside_output = seeded_result(197.0);
    const ByteSnapshot<G1ClearanceResult> outside_before(outside_output);
    check(g1_sphere_clearance(
              outside_output, g1_pose_clearance_budget(),
              nonrepresentable, outside, radius,
              NULL, 0) == G1ClearanceOutsideDomain,
          "one-ULP excursion past non-binary32 maximum is outside");
    check(outside_before.same(outside_output),
          "non-binary32 outside failure is transactional");

    const float malformed_values[] = {
        float_from_bits(UINT32_C(0x7fc00001)),
        float_from_bits(UINT32_C(0x7f800000)),
        float_from_bits(UINT32_C(0xff800000)),
        float_from_bits(UINT32_C(0x00000001)),
        float_from_bits(UINT32_C(0x80000001)),
        float_from_bits(UINT32_C(0x80000000))
    };
    for (size_t malformed_index = 0;
         malformed_index < sizeof(malformed_values) /
                               sizeof(malformed_values[0]);
         ++malformed_index) {
        heightfield malformed;
        point_make_field(malformed, 5, 5, 0.0f, 0.0f, 0.25f);
        malformed.heights(2 + 2 * malformed.nx) =
            malformed_values[malformed_index];
        G1ClearanceResult malformed_sphere = seeded_result(199.0);
        G1ClearanceResult malformed_capsule = seeded_result(201.0);
        const ByteSnapshot<G1ClearanceResult> sphere_before(
            malformed_sphere);
        const ByteSnapshot<G1ClearanceResult> capsule_before(
            malformed_capsule);
        check(g1_sphere_clearance(
                  malformed_sphere, g1_pose_clearance_budget(), malformed,
                  vec3(0.5f, 1.0f, 0.5f), 0.25f,
                  NULL, 0) == G1ClearanceInvalidField &&
              g1_capsule_clearance(
                  malformed_capsule, g1_pose_clearance_budget(), malformed,
                  vec3(0.375f, 1.0f, 0.5f),
                  vec3(0.625f, 1.25f, 0.5f), 0.25f,
                  NULL, 0) == G1ClearanceInvalidField,
              "visited malformed terrain height is invalid-field");
        check(sphere_before.same(malformed_sphere) &&
                  capsule_before.same(malformed_capsule),
              "malformed-height failures are transactional");
    }

    heightfield clean_unvisited;
    point_make_field(clean_unvisited, 5, 5,
                     0.0f, 0.0f, 0.25f);
    heightfield poisoned_unvisited = clean_unvisited;
    poisoned_unvisited.heights(4 + 4 * poisoned_unvisited.nx) =
        float_from_bits(UINT32_C(0x7fc00001));
    const vec3 local_center(0.25f, 1.0f, 0.25f);
    const vec3 local_upper(0.25f, 1.25f, 0.25f);
    G1ClearanceResult clean_sphere = seeded_result(203.0);
    G1ClearanceResult poisoned_sphere = seeded_result(205.0);
    G1ClearanceResult clean_capsule = seeded_result(207.0);
    G1ClearanceResult poisoned_capsule = seeded_result(209.0);
    check(g1_sphere_clearance(
              clean_sphere, g1_pose_clearance_budget(), clean_unvisited,
              local_center, 0.1f, NULL, 0) == G1ClearanceOk &&
          g1_sphere_clearance(
              poisoned_sphere, g1_pose_clearance_budget(),
              poisoned_unvisited,
              local_center, 0.1f, NULL, 0) == G1ClearanceOk &&
          g1_capsule_clearance(
              clean_capsule, g1_pose_clearance_budget(), clean_unvisited,
              local_center, local_upper, 0.1f,
              NULL, 0) == G1ClearanceOk &&
          g1_capsule_clearance(
              poisoned_capsule, g1_pose_clearance_budget(),
              poisoned_unvisited,
              local_upper, local_center, 0.1f,
              NULL, 0) == G1ClearanceOk,
          "unvisited malformed height does not invalidate a primitive");
    check(clearance_result_same(clean_sphere, poisoned_sphere) &&
              clearance_result_same(clean_capsule, poisoned_capsule),
          "unvisited malformed height cannot influence certified output");
}

static heightfield task34_make_count_field(int cells_x)
{
    heightfield field;
    point_make_field(field, cells_x + 1, 2,
                     0.0f, 0.0f, 1.0f);
    field.heights.set(std::numeric_limits<float>::quiet_NaN());
    return field;
}

static void task34_require_capsule_budget_failure(
    const heightfield& field,
    vec3 endpoint_b,
    const G1ClearanceBudget& limits,
    G1ClearanceStatus expected,
    const char* message)
{
    G1ClearanceResult output = seeded_result(211.0);
    const ByteSnapshot<G1ClearanceResult> before(output);
    check(g1_capsule_clearance(
              output, limits, field,
              vec3(0.25f, 1.0f, 0.5f), endpoint_b, 0.25f,
              NULL, 0) == expected,
          message);
    check(before.same(output),
          "count-preflight failure preserves capsule output");
}

static void test_task34_rectangular_count_preflights()
{
    const heightfield exact = task34_make_count_field(512);
    const vec3 exact_end(511.75f, 1.0f, 0.5f);
    task34_require_capsule_budget_failure(
        exact, exact_end, g1_pose_clearance_budget(),
        G1ClearanceInvalidField,
        "exact 512-cell/1024-pair capsule proceeds to its first height load");

    G1ClearanceBudget tightened = g1_pose_clearance_budget();
    tightened.maximum_cells = 511;
    task34_require_capsule_budget_failure(
        exact, exact_end, tightened, G1ClearanceBudgetExceeded,
        "511-cell caller budget rejects 512 cells before terrain load");
    tightened = g1_pose_clearance_budget();
    tightened.maximum_primitive_triangle_pairs = 1023;
    task34_require_capsule_budget_failure(
        exact, exact_end, tightened, G1ClearanceBudgetExceeded,
        "1023-pair caller budget rejects 1024 pairs before terrain load");
    tightened = g1_pose_clearance_budget();
    tightened.maximum_face_patches = 8191;
    task34_require_capsule_budget_failure(
        exact, exact_end, tightened, G1ClearanceBudgetExceeded,
        "8191-patch caller budget rejects fixed patch work before load");
    tightened = g1_pose_clearance_budget();
    tightened.maximum_candidate_tests = 32767;
    task34_require_capsule_budget_failure(
        exact, exact_end, tightened, G1ClearanceBudgetExceeded,
        "32767-candidate budget rejects fixed analytic work before load");

    const heightfield over = task34_make_count_field(513);
    task34_require_capsule_budget_failure(
        over, vec3(512.75f, 1.0f, 0.5f),
        g1_pose_clearance_budget(), G1ClearanceBudgetExceeded,
        "513-cell/1026-pair primitive cap rejects before terrain load");
}

static void test_task34_named_tent_sign_reversal()
{
    const float cell = float_from_bits(UINT32_C(0x3b1efa48));
    heightfield tent;
    point_make_field(
        tent, 5, 5,
        float_from_bits(UINT32_C(0xbb8b1b00)),
        float_from_bits(UINT32_C(0xbb9ef53c)),
        cell);
    tent.heights(2 + 2 * tent.nx) =
        float_from_bits(UINT32_C(0x3b83126f));
    const float radius = float_from_bits(UINT32_C(0x381efa48));
    const float center_y = float_from_bits(UINT32_C(0x3b543300));
    const vec3 start(0.0f, center_y, 0.0f);
    const vec3 stop(
        float_from_bits(UINT32_C(0x3b6e7765)), center_y,
        float_from_bits(UINT32_C(0x36723088)));
    const volatile float spacing = 0.5f * cell;
    const volatile float dx = stop.x - start.x;
    const volatile float dz = stop.z - start.z;
    const volatile float length = std::sqrt(dx * dx + dz * dz);
    check(static_cast<float>(length / spacing) == 3.0f,
          "named tent reproduces the removed three-step float lattice");
    const int segment_steps = static_cast<int>(
        std::ceil(static_cast<float>(length / spacing)));
    check(segment_steps == 3,
          "named tent old lattice has exactly four centerline samples");
    vec3 old_samples[4] = {};
    const uint32_t expected_sample_bits[4][2] = {
        {UINT32_C(0x00000000), UINT32_C(0x00000000)},
        {UINT32_C(0x3a9efa44), UINT32_C(0x35a175b0)},
        {UINT32_C(0x3b1efa44), UINT32_C(0x362175b0)},
        {UINT32_C(0x3b6e7765), UINT32_C(0x36723088)}
    };
    for (int step = 0; step <= segment_steps; ++step) {
        const volatile float parameter =
            static_cast<float>(step) /
            static_cast<float>(segment_steps);
        const volatile float sample_x =
            start.x + parameter * dx;
        const volatile float sample_z =
            start.z + parameter * dz;
        old_samples[step] = vec3(sample_x, center_y, sample_z);
        check(float_bits(old_samples[step].x) ==
                  expected_sample_bits[step][0] &&
              float_bits(old_samples[step].z) ==
                  expected_sample_bits[step][1],
              "named tent regenerates each removed float-lattice sample");
    }
    const double half_cell =
        0.5 * static_cast<double>(cell);
    for (int gap_index = 0; gap_index < 2; ++gap_index) {
        const double gap_x =
            static_cast<double>(old_samples[gap_index + 1].x) -
            static_cast<double>(old_samples[gap_index].x);
        const double gap_z =
            static_cast<double>(old_samples[gap_index + 1].z) -
            static_cast<double>(old_samples[gap_index].z);
        const double gap = std::sqrt(
            gap_x * gap_x + gap_z * gap_z);
        check(gap > half_cell,
              "named tent first two rounded gaps exceed half a cell");
    }

    const int radial_steps = static_cast<int>(
        std::ceil(radius / spacing));
    int admitted_radial_offsets = 0;
    for (int radial_z = -radial_steps;
         radial_z <= radial_steps;
         ++radial_z) {
        for (int radial_x = -radial_steps;
             radial_x <= radial_steps;
             ++radial_x) {
            const volatile float offset_x =
                static_cast<float>(radial_x) * spacing;
            const volatile float offset_z =
                static_cast<float>(radial_z) * spacing;
            const volatile float offset_square =
                offset_x * offset_x + offset_z * offset_z;
            if (offset_square <= radius * radius) {
                ++admitted_radial_offsets;
            }
        }
    }
    check(admitted_radial_offsets == 1,
          "named tent radial lattice admits only its center offset");

    double old_minimum = std::numeric_limits<double>::infinity();
    for (int step = 0; step <= segment_steps; ++step) {
        for (int radial_z = -radial_steps;
             radial_z <= radial_steps;
             ++radial_z) {
            for (int radial_x = -radial_steps;
                 radial_x <= radial_steps;
                 ++radial_x) {
                const volatile float offset_x =
                    static_cast<float>(radial_x) * spacing;
                const volatile float offset_z =
                    static_cast<float>(radial_z) * spacing;
                const volatile float offset_square =
                    offset_x * offset_x + offset_z * offset_z;
                const volatile float radius_square = radius * radius;
                if (offset_square > radius_square) {
                    continue;
                }
                G1SurfaceSample sample = {};
                check(g1_surface_query_v2(
                          sample, tent,
                          old_samples[step].x + offset_x,
                          old_samples[step].z + offset_z) ==
                          G1SurfaceQueryValid,
                      "named tent old-lattice sample is in-domain");
                const double vertical_radius = std::sqrt(
                    static_cast<double>(radius_square) -
                    static_cast<double>(offset_square));
                const double clearance =
                    static_cast<double>(center_y) -
                    vertical_radius -
                    static_cast<double>(sample.height);
                old_minimum = clearance < old_minimum
                    ? clearance
                    : old_minimum;
            }
        }
    }
    check(old_minimum > 0.00019,
          "removed centerline/radial lattice falsely clears the named tent");

    G1ClearanceResult output = seeded_result(223.0);
    check(g1_capsule_clearance(
              output, g1_pose_clearance_budget(), tent,
              start, stop, radius, NULL, 0) == G1ClearanceOk,
          "named tent capsule must certify continuously");
    check(output.witness_upper_m < -0.00079 &&
              output.witness_upper_m - output.lower_bound_m <=
                  G1ClearanceMaximumCertificateWidthM,
          "certified named-tent witness reverses the old positive sign");
    G1ClearanceResult reverse = seeded_result(224.0);
    check(g1_capsule_clearance(
              reverse, g1_pose_clearance_budget(), tent,
              stop, start, radius, NULL, 0) == G1ClearanceOk &&
          clearance_result_same(output, reverse),
          "named-tent certificate is bit-identical under reversal");
}

static void task34_require_sphere_capsule_guard(
    const heightfield& field,
    vec3 center,
    float radius,
    double minimum_guard)
{
    G1SurfaceSample producer = {};
    check(g1_surface_query_v2(
              producer, field, center.x, center.z) ==
              G1SurfaceQueryValid,
          "sphere/capsule guard probe has a valid producer sample");
    const vec3 upper_endpoint(
        center.x, center.y + 0.25f, center.z);
    G1ClearanceResult sphere = seeded_result(225.0);
    G1ClearanceResult capsule = seeded_result(226.0);
    G1ClearanceResult reverse = seeded_result(227.0);
    check(g1_sphere_clearance(
              sphere, g1_pose_clearance_budget(), field,
              center, radius, NULL, 0) == G1ClearanceOk &&
          g1_capsule_clearance(
              capsule, g1_pose_clearance_budget(), field,
              center, upper_endpoint, radius,
              NULL, 0) == G1ClearanceOk &&
          g1_capsule_clearance(
              reverse, g1_pose_clearance_budget(), field,
              upper_endpoint, center, radius,
              NULL, 0) == G1ClearanceOk,
          "sphere/capsule output-guard fixtures certify");
    const double producer_clearance =
        static_cast<double>(center.y) -
        static_cast<double>(radius) -
        static_cast<double>(producer.height);
    check(sphere.lower_bound_m <= producer_clearance &&
              capsule.lower_bound_m <= producer_clearance,
          "sphere/capsule lower bounds preserve producer-height safety");
    check(sphere.witness_upper_m - sphere.lower_bound_m >=
              minimum_guard &&
              capsule.witness_upper_m - capsule.lower_bound_m >=
              minimum_guard,
          "sphere/capsule certificates retain the mandatory float guard");
    check(clearance_result_same(capsule, reverse),
          "guard capsule remains bit-identical under endpoint reversal");
}

static void test_task34_sphere_capsule_output_guards()
{
    heightfield unit;
    point_make_field(unit, 5, 5, 0.0f, 0.0f, 0.25f);
    unit.heights.set(1.0f);
    const vec3 center(0.5f, 2.0f, 0.5f);
    const float radius = 0.125f;
    G1SurfaceSample producer = {};
    check(g1_surface_query_v2(
              producer, unit, center.x, center.z) ==
              G1SurfaceQueryValid,
          "sphere guard fixture has a valid center producer sample");
    G1ClearanceResult sphere = seeded_result(227.0);
    G1ClearanceResult capsule = seeded_result(229.0);
    check(g1_sphere_clearance(
              sphere, g1_pose_clearance_budget(), unit,
              center, radius, NULL, 0) == G1ClearanceOk &&
          g1_capsule_clearance(
              capsule, g1_pose_clearance_budget(), unit,
              center, center, radius, NULL, 0) == G1ClearanceOk,
          "unit-height sphere/capsule guard fixtures certify");
    const double producer_clearance =
        static_cast<double>(center.y) -
        static_cast<double>(radius) -
        static_cast<double>(producer.height);
    check(sphere.lower_bound_m <= producer_clearance &&
              capsule.lower_bound_m <= producer_clearance,
          "sphere/capsule lower bounds preserve producer-height safety");

    const float one_up =
        float_from_bits(float_bits(1.0f) + 1);
    const double one_ulp =
        static_cast<double>(one_up) - 1.0;
    heightfield upward;
    point_make_field(upward);
    upward.heights(0) = 1.0f;
    upward.heights(1) = one_up;
    upward.heights(2) = 1.0f;
    upward.heights(3) = one_up;
    const vec3 upward_center(0.75f, 2.0f, 0.25f);
    G1SurfaceSample upward_sample = {};
    check(g1_surface_query_v2(
              upward_sample, upward,
              upward_center.x, upward_center.z) ==
              G1SurfaceQueryValid &&
          float_bits(upward_sample.height) == float_bits(one_up),
          "sphere/capsule fixture forces upward binary32 rounding");
    task34_require_sphere_capsule_guard(
        upward, upward_center, 0.125f, one_ulp);

    const float minimum_normal =
        std::numeric_limits<float>::min();
    const float zero_band_signs[] = {
        -minimum_normal, minimum_normal
    };
    for (size_t sign = 0;
         sign < sizeof(zero_band_signs) /
                    sizeof(zero_band_signs[0]);
         ++sign) {
        heightfield zero_band;
        point_make_field(zero_band);
        zero_band.heights(0) = zero_band_signs[sign];
        zero_band.heights(1) = 0.0f;
        zero_band.heights(2) = zero_band_signs[sign];
        zero_band.heights(3) = 0.0f;
        const vec3 zero_center(0.5f, 1.0f, 0.25f);
        G1SurfaceSample zero_sample = {};
        check(g1_surface_query_v2(
                  zero_sample, zero_band,
                  zero_center.x, zero_center.z) ==
                  G1SurfaceQueryValid &&
              float_bits(zero_sample.height) == 0,
              "sphere/capsule zero-band producer canonicalizes to +zero");
        task34_require_sphere_capsule_guard(
            zero_band, zero_center, 0.125f,
            static_cast<double>(minimum_normal));
    }

    const float binades[] = {1.0f, 2.0f};
    const vec3 side_centers[] = {
        vec3(0.75f, 4.0f, 0.25f),
        vec3(0.25f, 4.0f, 0.75f)
    };
    for (size_t binade = 0;
         binade < sizeof(binades) / sizeof(binades[0]);
         ++binade) {
        const float boundary = binades[binade];
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
             side < sizeof(side_centers) /
                        sizeof(side_centers[0]);
             ++side) {
            task34_require_sphere_capsule_guard(
                crossing, side_centers[side], 0.125f,
                larger_ulp);
        }
    }

    heightfield flat_sixteen;
    point_make_field(flat_sixteen, 5, 5,
                     0.0f, 0.0f, 0.25f);
    flat_sixteen.heights.set(16.0f);
    G1ClearanceResult failed_sphere = seeded_result(233.0);
    G1ClearanceResult failed_capsule = seeded_result(239.0);
    const ByteSnapshot<G1ClearanceResult> sphere_before(failed_sphere);
    const ByteSnapshot<G1ClearanceResult> capsule_before(failed_capsule);
    check(g1_sphere_clearance(
              failed_sphere, g1_pose_clearance_budget(), flat_sixteen,
              vec3(0.5f, 20.0f, 0.5f), radius,
              NULL, 0) == G1ClearanceUncertified,
          "flat-16 sphere fails closed on mandatory output guard");
    check(g1_capsule_clearance(
              failed_capsule, g1_pose_clearance_budget(), flat_sixteen,
              vec3(0.5f, 20.0f, 0.5f),
              vec3(0.75f, 20.0f, 0.75f), radius,
              NULL, 0) == G1ClearanceUncertified,
          "flat-16 capsule fails closed on mandatory output guard");
    check(sphere_before.same(failed_sphere) &&
              capsule_before.same(failed_capsule),
          "mandatory-guard failures preserve sphere/capsule outputs");
}

static void task34_make_kind2_winner_fixture(
    heightfield& field,
    vec3& endpoint_a,
    vec3& endpoint_b,
    float& radius)
{
    point_make_field(
        field, 3, 3, 0.0f, 0.0f,
        float_from_bits(UINT32_C(0x3e800000)));
    const uint32_t height_bits[] = {
        UINT32_C(0x3f200000), UINT32_C(0x3f200000),
        UINT32_C(0xbf700000), UINT32_C(0xbe000000),
        UINT32_C(0xbf300000), UINT32_C(0x3f200000),
        UINT32_C(0x3e400000), UINT32_C(0x3ee00000),
        UINT32_C(0xbee00000)
    };
    for (int index = 0; index < 9; ++index) {
        field.heights(index) = float_from_bits(height_bits[index]);
    }
    endpoint_a = vec3(
        float_from_bits(UINT32_C(0x3d800000)),
        float_from_bits(UINT32_C(0x40140000)),
        float_from_bits(UINT32_C(0x3e200000)));
    endpoint_b = vec3(
        float_from_bits(UINT32_C(0x3e800000)),
        float_from_bits(UINT32_C(0x3ee00000)),
        float_from_bits(UINT32_C(0x3d800000)));
    radius = float_from_bits(UINT32_C(0x3d800000));
}

static void test_task34_exact_dyadic_fallback()
{
    heightfield field;
    point_make_field(field, 3, 3, 0.0f, 0.0f, 0.25f);
    const uint32_t height_bits[] = {
        UINT32_C(0x3ec00000), UINT32_C(0x3e000000),
        UINT32_C(0x3fe00000), UINT32_C(0x3fa00000),
        UINT32_C(0x3e800000), UINT32_C(0x3e000000),
        UINT32_C(0x3fb00000), UINT32_C(0x3f900000),
        UINT32_C(0x3fe00000)
    };
    for (int index = 0; index < 9; ++index) {
        field.heights(index) = float_from_bits(height_bits[index]);
    }
    const vec3 endpoint_a(
        float_from_bits(UINT32_C(0x3ed00000)),
        float_from_bits(UINT32_C(0x40100000)),
        float_from_bits(UINT32_C(0x3e620000)));
    const vec3 endpoint_b(
        float_from_bits(UINT32_C(0x3ec60000)),
        float_from_bits(UINT32_C(0x3fc80000)),
        float_from_bits(UINT32_C(0x3e4e0000)));
    const float radius = float_from_bits(UINT32_C(0x3dc00000));

    G1ClearanceResult output = seeded_result(241.0);
    check(g1_capsule_clearance(
              output, g1_pose_clearance_budget(), field,
              endpoint_a, endpoint_b, radius,
              NULL, 0) == G1ClearanceOk,
          "adversarial capsule closes through exact-dyadic fallback");
    check(output.lower_bound_m <= output.witness_upper_m &&
              output.witness_upper_m - output.lower_bound_m <=
                  G1ClearanceMaximumCertificateWidthM,
          "fallback capsule returns the required certified width");
    const long double oracle =
        1.5625L -
        (6.5L * 0.38671875L -
         6.5L * 0.201171875L - 1.5L) -
        0.09375L * std::sqrt(85.5L);
    check(static_cast<long double>(output.lower_bound_m) <= oracle &&
              oracle <=
                  static_cast<long double>(output.witness_upper_m),
          "fallback capsule encloses the independent B-cap plane oracle");
    check(output.work.subdivision_nodes == 38,
          "fallback capsule charges its deterministic dyadic child count");

    G1ClearanceResult reverse = seeded_result(247.0);
    check(g1_capsule_clearance(
              reverse, g1_pose_clearance_budget(), field,
              endpoint_b, endpoint_a, radius,
              NULL, 0) == G1ClearanceOk &&
          clearance_result_same(output, reverse),
          "exact-dyadic fallback is bit-identical under endpoint reversal");

    G1ClearanceBudget no_subdivision = g1_pose_clearance_budget();
    no_subdivision.maximum_subdivision_nodes = 0;
    G1ClearanceResult failed = seeded_result(251.0);
    const ByteSnapshot<G1ClearanceResult> failed_before(failed);
    check(g1_capsule_clearance(
              failed, no_subdivision, field,
              endpoint_a, endpoint_b, radius,
              NULL, 0) == G1ClearanceUncertified,
          "zero subdivision cap fails closed on an unresolved patch");
    check(failed_before.same(failed),
          "exhausted fallback cap preserves capsule output");

    G1ClearanceBudget exact_subdivision = g1_pose_clearance_budget();
    exact_subdivision.maximum_subdivision_nodes = 38;
    G1ClearanceResult exact = seeded_result(253.0);
    check(g1_capsule_clearance(
              exact, exact_subdivision, field,
              endpoint_a, endpoint_b, radius,
              NULL, 0) == G1ClearanceOk &&
          clearance_result_same(output, exact),
          "exact 38-child fallback cap reproduces the default result");

    G1ClearanceBudget one_child_short = g1_pose_clearance_budget();
    one_child_short.maximum_subdivision_nodes = 37;
    G1ClearanceResult short_output = seeded_result(257.0);
    const ByteSnapshot<G1ClearanceResult> short_before(short_output);
    check(g1_capsule_clearance(
              short_output, one_child_short, field,
              endpoint_a, endpoint_b, radius,
              NULL, 0) == G1ClearanceUncertified,
          "odd cap below the atomic 38-child work fails closed");
    check(short_before.same(short_output),
          "one-child-short fallback cap preserves capsule output");
}

static void test_task34_public_kind2_winner()
{
    heightfield field;
    vec3 endpoint_a;
    vec3 endpoint_b;
    float radius = 0.0f;
    task34_make_kind2_winner_fixture(
        field, endpoint_a, endpoint_b, radius);

    G1ClearanceResult forward = seeded_result(261.0);
    G1ClearanceResult reverse = seeded_result(263.0);
    check(g1_capsule_clearance(
              forward, g1_pose_clearance_budget(), field,
              endpoint_a, endpoint_b, radius,
              NULL, 0) == G1ClearanceOk &&
          g1_capsule_clearance(
              reverse, g1_pose_clearance_budget(), field,
              endpoint_b, endpoint_a, radius,
              NULL, 0) == G1ClearanceOk &&
          clearance_result_same(forward, reverse),
          "public kind-2 winner is bit-identical under endpoint reversal");
    check(double_bits(forward.lower_bound_m) ==
              UINT64_C(0xbfcda82ffb3e5db8) &&
          double_bits(forward.witness_upper_m) ==
              UINT64_C(0xbfcda827999fceea) &&
          forward.witness.primitive_index == 0 &&
          forward.witness.cell_x == 1 &&
          forward.witness.cell_z == 0 &&
          forward.witness.terrain_triangle_index == 1 &&
          forward.witness.patch_index == 3 &&
          forward.witness.candidate_kind == 2 &&
          forward.witness.candidate_subindex == 35,
          "public kind-2 winner pins bounds and immutable creation ordinal key");
    const G1ClearanceWork expected_work = {
        0, 2, 4, 32, 128, 5622
    };
    check(clearance_work_same(forward.work, expected_work),
          "public kind-2 winner pins deterministic fallback work");
    check(double_bits(forward.witness.body_x) ==
              UINT64_C(0x3fd2000000000000) &&
          double_bits(forward.witness.body_y) ==
              UINT64_C(0x3fd92bec33301888) &&
          double_bits(forward.witness.body_z) ==
              UINT64_C(0x3fa0000000000000) &&
          double_bits(forward.witness.surface_x) ==
              UINT64_C(0x3fd2000000000000) &&
          double_bits(forward.witness.surface_y) ==
              UINT64_C(0x3fe4000000000000) &&
          double_bits(forward.witness.surface_z) ==
              UINT64_C(0x3fa0000000000000) &&
          double_bits(forward.witness.segment_parameter) ==
              UINT64_C(0x3ff0000000000000) &&
          double_bits(forward.witness.terrain_weight_0) ==
              UINT64_C(0x3fec000000000000) &&
          double_bits(forward.witness.terrain_weight_1) ==
              UINT64_C(0x3fc0000000000000) &&
          double_bits(forward.witness.terrain_weight_2) ==
              UINT64_C(0x0000000000000000),
          "public kind-2 winner pins complete dyadic witness diagnostics");

    const long double radius_ld = static_cast<long double>(radius);
    const long double oracle =
        0.4375L - 0.625L -
        std::sqrt(radius_ld * radius_ld -
                  2.0L * 0.03125L * 0.03125L);
    check(static_cast<long double>(forward.lower_bound_m) <= oracle &&
              oracle <=
                  static_cast<long double>(forward.witness_upper_m),
          "public kind-2 winner encloses its independently reconstructed oracle");
    task34_require_public_witness_enclosure(
        field, endpoint_a, endpoint_b, radius, forward);
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

static bool task5_witness_key_less(
    const G1ClearanceWitness& left,
    const G1ClearanceWitness& right)
{
    if (left.primitive_index != right.primitive_index) {
        return left.primitive_index < right.primitive_index;
    }
    if (left.cell_z != right.cell_z) {
        return left.cell_z < right.cell_z;
    }
    if (left.cell_x != right.cell_x) {
        return left.cell_x < right.cell_x;
    }
    if (left.terrain_triangle_index !=
        right.terrain_triangle_index) {
        return left.terrain_triangle_index <
               right.terrain_triangle_index;
    }
    if (left.patch_index != right.patch_index) {
        return left.patch_index < right.patch_index;
    }
    if (left.candidate_kind != right.candidate_kind) {
        return left.candidate_kind < right.candidate_kind;
    }
    return left.candidate_subindex < right.candidate_subindex;
}

static void task5_add_work(
    G1ClearanceWork& total,
    const G1ClearanceWork& addend)
{
    uint32_t G1ClearanceWork::* const members[] = {
        &G1ClearanceWork::point_queries,
        &G1ClearanceWork::cells_visited,
        &G1ClearanceWork::primitive_triangle_pairs,
        &G1ClearanceWork::face_patches,
        &G1ClearanceWork::candidate_tests,
        &G1ClearanceWork::subdivision_nodes
    };
    for (const auto member : members) {
        const uint64_t sum =
            static_cast<uint64_t>(total.*member) +
            static_cast<uint64_t>(addend.*member);
        check(sum <= UINT32_MAX,
              "Task 5 independent work sum fits uint32");
        total.*member = static_cast<uint32_t>(sum);
    }
}

static G1ClearanceResult task5_aggregate_results(
    const G1ClearanceResult* results,
    int count)
{
    check(results != NULL && count > 0,
          "Task 5 aggregate oracle has members");
    const G1ClearanceResult* lower = &results[0];
    const G1ClearanceResult* witness = &results[0];
    G1ClearanceWork total = {};
    for (int index = 0; index < count; ++index) {
        if (results[index].lower_bound_m < lower->lower_bound_m) {
            lower = &results[index];
        }
        if (results[index].witness_upper_m <
                witness->witness_upper_m ||
            (double_bits(results[index].witness_upper_m) ==
                 double_bits(witness->witness_upper_m) &&
             task5_witness_key_less(
                 results[index].witness, witness->witness))) {
            witness = &results[index];
        }
        task5_add_work(total, results[index].work);
    }
    G1ClearanceResult output = *lower;
    output.witness_upper_m = witness->witness_upper_m;
    output.witness = witness->witness;
    output.work = total;
    return output;
}

static G1ClearanceResult task5_rekey_result(
    G1ClearanceResult result,
    uint32_t primitive_index)
{
    result.witness.primitive_index = primitive_index;
    return result;
}

static G1ClearanceResult task5_require_point(
    const heightfield& field,
    vec3 point,
    uint32_t primitive_index)
{
    G1ClearanceResult result = seeded_result(301.0 + primitive_index);
    char error[256] = {};
    check(g1_point_clearance(
              result, g1_pose_clearance_budget(), field, point,
              error, static_cast<int>(sizeof(error))) == G1ClearanceOk,
          error[0] == '\0' ?
              "Task 5 direct point certifies" : error);
    return task5_rekey_result(result, primitive_index);
}

static G1ClearanceResult task5_require_sphere(
    const heightfield& field,
    vec3 center,
    float radius,
    uint32_t primitive_index,
    const G1ClearanceBudget& limits = g1_pose_clearance_budget())
{
    G1ClearanceResult result = seeded_result(401.0 + primitive_index);
    char error[256] = {};
    check(g1_sphere_clearance(
              result, limits, field, center, radius,
              error, static_cast<int>(sizeof(error))) == G1ClearanceOk,
          error[0] == '\0' ?
              "Task 5 direct sphere certifies" : error);
    return task5_rekey_result(result, primitive_index);
}

static G1ClearanceResult task5_require_capsule(
    const heightfield& field,
    vec3 endpoint_a,
    vec3 endpoint_b,
    float radius,
    uint32_t primitive_index,
    const G1ClearanceBudget& limits = g1_pose_clearance_budget())
{
    G1ClearanceResult result = seeded_result(501.0 + primitive_index);
    char error[256] = {};
    check(g1_capsule_clearance(
              result, limits, field,
              endpoint_a, endpoint_b, radius,
              error, static_cast<int>(sizeof(error))) == G1ClearanceOk,
          error[0] == '\0' ?
              "Task 5 direct capsule certifies" : error);
    return task5_rekey_result(result, primitive_index);
}

static G1ClearanceResult task5_direct_foot(
    const heightfield& field,
    const vec3 centers[4],
    float radius,
    uint32_t primitive_base,
    const G1ClearanceBudget& limits = g1_pose_clearance_budget())
{
    G1ClearanceResult spheres[4] = {};
    for (uint32_t index = 0; index < 4; ++index) {
        spheres[index] = task5_require_sphere(
            field, centers[index], radius,
            primitive_base + index, limits);
    }
    return task5_aggregate_results(spheres, 4);
}

static G1ClearanceResult task5_direct_swept_foot(
    const heightfield& field,
    const vec3 previous[4],
    const vec3 current[4],
    float radius,
    uint32_t primitive_base)
{
    const G1ClearanceBudget limits =
        g1_swing_foot_clearance_budget();
    G1ClearanceResult capsules[4] = {};
    for (uint32_t index = 0; index < 4; ++index) {
        capsules[index] = task5_require_capsule(
            field, previous[index], current[index], radius,
            primitive_base + index, limits);
    }
    return task5_aggregate_results(capsules, 4);
}

struct Task5LockedLegGeometry
{
    vec3 foot_centers[4];
    vec3 thigh_a;
    vec3 thigh_b;
    vec3 shin_a;
    vec3 shin_b;
};

#ifndef __FAST_MATH__
static bool task5_vec3_bits_same(vec3 left, vec3 right)
{
    return float_bits(left.x) == float_bits(right.x) &&
           float_bits(left.y) == float_bits(right.y) &&
           float_bits(left.z) == float_bits(right.z);
}
#endif

static Task5LockedLegGeometry task5_locked_leg_geometry()
{
    Task5LockedLegGeometry geometry = {};
    geometry.foot_centers[0] = vec3(
        float_from_bits(UINT32_C(0xbe4ccccd)),
        float_from_bits(UINT32_C(0x3f051eb9)),
        float_from_bits(UINT32_C(0xbccccccd)));
    geometry.foot_centers[1] = vec3(
        float_from_bits(UINT32_C(0xbe4ccccd)),
        float_from_bits(UINT32_C(0x3f051eb9)),
        float_from_bits(UINT32_C(0xbd99999a)));
    geometry.foot_centers[2] = vec3(
        float_from_bits(UINT32_C(0xbebd70a4)),
        float_from_bits(UINT32_C(0x3f051eb9)),
        float_from_bits(UINT32_C(0xbca3d70b)));
    geometry.foot_centers[3] = vec3(
        float_from_bits(UINT32_C(0xbebd70a4)),
        float_from_bits(UINT32_C(0x3f051eb9)),
        float_from_bits(UINT32_C(0xbda3d70a)));
    geometry.thigh_a = vec3(
        float_from_bits(UINT32_C(0xbeb33333)),
        float_from_bits(UINT32_C(0x3f9c28f6)),
        float_from_bits(UINT32_C(0xbe19999a)));
    geometry.thigh_b = vec3(
        float_from_bits(UINT32_C(0xbedb22d1)),
        float_from_bits(UINT32_C(0x3faf5c29)),
        float_from_bits(UINT32_C(0xbe19999a)));
    geometry.shin_a = vec3(
        float_from_bits(UINT32_C(0xbe99999a)),
        float_from_bits(UINT32_C(0x3f733333)),
        float_from_bits(UINT32_C(0xbdcccccd)));
    geometry.shin_b = vec3(
        float_from_bits(UINT32_C(0xbe99999a)),
        float_from_bits(UINT32_C(0x3f970a3d)),
        float_from_bits(UINT32_C(0xbdcccccd)));
    return geometry;
}

static void task5_check_strict_transform_oracle(
    const slice1d<vec3> global_positions,
    const slice1d<quat> global_rotations,
    const G1LegConfig& config,
    const Task5LockedLegGeometry& locked)
{
#ifndef __FAST_MATH__
    for (int index = 0; index < 4; ++index) {
        const vec3 transformed =
            global_positions(config.ankle) +
            quat_mul_vec3(
                global_rotations(config.ankle),
                config.foot_sphere_centers_local[index]);
        check(task5_vec3_bits_same(
                  transformed, locked.foot_centers[index]),
              "Task 5 locked foot transform bits match strict formula");
    }
    const vec3 transformed_thigh_a =
        global_positions(config.hip) +
        quat_mul_vec3(
            global_rotations(config.hip), config.thigh_start_local);
    const vec3 transformed_thigh_b =
        global_positions(config.hip) +
        quat_mul_vec3(
            global_rotations(config.hip), config.thigh_end_local);
    const vec3 transformed_shin_a =
        global_positions(config.knee) +
        quat_mul_vec3(
            global_rotations(config.knee), config.shin_start_local);
    const vec3 transformed_shin_b =
        global_positions(config.knee) +
        quat_mul_vec3(
            global_rotations(config.knee), config.shin_end_local);
    check(task5_vec3_bits_same(transformed_thigh_a, locked.thigh_a) &&
              task5_vec3_bits_same(transformed_thigh_b, locked.thigh_b) &&
              task5_vec3_bits_same(transformed_shin_a, locked.shin_a) &&
              task5_vec3_bits_same(transformed_shin_b, locked.shin_b),
          "Task 5 locked limb transform bits match strict formulas");
#else
    (void)global_positions;
    (void)global_rotations;
    (void)config;
    (void)locked;
#endif
}

static G1LegClearance task5_direct_leg(
    const heightfield& field,
    const slice1d<vec3> global_positions,
    const slice1d<quat> global_rotations,
    const G1LegConfig& config,
    uint32_t primitive_base)
{
    G1LegClearance output = {};
    output.knee = task5_require_point(
        field, global_positions(config.knee), primitive_base + 0);
    output.ankle = task5_require_point(
        field, global_positions(config.ankle), primitive_base + 1);
    output.toe = task5_require_point(
        field, global_positions(config.contact), primitive_base + 2);

    const Task5LockedLegGeometry geometry =
        task5_locked_leg_geometry();
    task5_check_strict_transform_oracle(
        global_positions, global_rotations, config, geometry);
    output.foot = task5_direct_foot(
        field, geometry.foot_centers, config.foot_sphere_radius_m,
        primitive_base + 3);

    output.thigh = task5_require_capsule(
        field, geometry.thigh_a, geometry.thigh_b,
        config.thigh_radius_m,
        primitive_base + 7);

    output.shin = task5_require_capsule(
        field, geometry.shin_a, geometry.shin_b,
        config.shin_radius_m,
        primitive_base + 8);

    const G1ClearanceResult components[] = {
        output.knee,
        output.ankle,
        output.toe,
        output.foot,
        output.thigh,
        output.shin
    };
    output.minimum = task5_aggregate_results(
        components,
        static_cast<int>(
            sizeof(components) / sizeof(components[0])));
    return output;
}

static bool task5_leg_same(
    const G1LegClearance& left,
    const G1LegClearance& right)
{
    return clearance_result_same(left.knee, right.knee) &&
           clearance_result_same(left.ankle, right.ankle) &&
           clearance_result_same(left.toe, right.toe) &&
           clearance_result_same(left.foot, right.foot) &&
           clearance_result_same(left.thigh, right.thigh) &&
           clearance_result_same(left.shin, right.shin) &&
           clearance_result_same(left.minimum, right.minimum);
}

static G1LegClearance task5_seed_leg(double base)
{
    G1LegClearance output = {};
    output.knee = seeded_result(base + 1.0);
    output.ankle = seeded_result(base + 2.0);
    output.toe = seeded_result(base + 3.0);
    output.foot = seeded_result(base + 4.0);
    output.thigh = seeded_result(base + 5.0);
    output.shin = seeded_result(base + 6.0);
    output.minimum = seeded_result(base + 7.0);
    return output;
}

static G1PoseClearance task5_seed_pose(double base)
{
    G1PoseClearance output = {};
    output.hips = seeded_result(base + 1.0);
    output.left = task5_seed_leg(base + 10.0);
    output.right = task5_seed_leg(base + 20.0);
    output.minimum = seeded_result(base + 30.0);
    return output;
}

static void task5_make_pose(
    vec3 positions[G1_BoneCount],
    quat rotations[G1_BoneCount])
{
    for (int bone = 0; bone < G1_BoneCount; ++bone) {
        positions[bone] = vec3(0.0f, 1.5f, 0.0f);
        rotations[bone] = quat();
    }
    positions[G1_Hips] = vec3(0.0f, 1.8f, 0.0f);

    const G1LegConfig left = g1_left_leg_config();
    positions[left.hip] = vec3(-0.35f, 1.2f, -0.15f);
    positions[left.knee] = vec3(-0.30f, 0.9f, -0.10f);
    positions[left.ankle] = vec3(-0.25f, 0.55f, -0.05f);
    positions[left.contact] = vec3(-0.10f, 0.70f, -0.05f);
    rotations[left.hip] = quat(0.0f, 1.0f, 0.0f, 0.0f);
    rotations[left.knee] = quat(0.0f, 0.0f, 0.0f, 1.0f);
    rotations[left.ankle] = quat(0.0f, 0.0f, 1.0f, 0.0f);

    const G1LegConfig right = g1_right_leg_config();
    positions[right.hip] = positions[left.hip];
    positions[right.knee] = positions[left.knee];
    positions[right.ankle] = positions[left.ankle];
    positions[right.contact] = positions[left.contact];
    rotations[right.hip] = quat(0.0f, 1.0f, 0.0f, 0.0f);
    rotations[right.knee] = quat(0.0f, 0.0f, 0.0f, 1.0f);
    rotations[right.ankle] = quat(0.0f, 0.0f, 1.0f, 0.0f);
}

static void test_task5_aggregate_certificates()
{
    heightfield field;
    point_make_field(field, 17, 17, -1.0f, -1.0f, 0.125f);

    const vec3 foot_centers[4] = {
        vec3(-0.45f, 0.90f, -0.20f),
        vec3(-0.15f, 1.10f, -0.10f),
        vec3(0.15f, 0.80f, 0.10f),
        vec3(0.45f, 0.80f, 0.20f)
    };
    const float radius = 0.02f;
    const G1ClearanceResult expected_foot = task5_direct_foot(
        field, foot_centers, radius, 0);
    G1ClearanceResult foot = seeded_result(601.0);
    char error[256] = {};
    check(g1_foot_clearance(
              foot, g1_pose_clearance_budget(), field,
              foot_centers, radius,
              error, static_cast<int>(sizeof(error))) == G1ClearanceOk,
          error[0] == '\0' ?
              "Task 5 foot aggregate certifies" : error);
    check(clearance_result_same(foot, expected_foot),
          "Task 5 foot equals four independent sphere certificates");
    check(foot.witness.primitive_index == 2,
          "Task 5 foot witness tie keeps the first stable primitive key");

    const vec3 previous[4] = {
        vec3(-0.45f, 0.90f, -0.20f),
        vec3(-0.15f, 1.00f, -0.10f),
        vec3(0.15f, 0.60f, 0.10f),
        vec3(0.45f, 1.10f, 0.20f)
    };
    const vec3 current[4] = {
        vec3(-0.42f, 0.91f, -0.18f),
        vec3(-0.12f, 1.01f, -0.08f),
        vec3(0.18f, float_from_bits(UINT32_C(0x3f000001)), 0.12f),
        vec3(0.48f, 1.11f, 0.22f)
    };
    const G1ClearanceResult expected_sweep = task5_direct_swept_foot(
        field, previous, current, radius, 0);
    G1ClearanceResult sweep = seeded_result(603.0);
    check(g1_swept_foot_clearance(
              sweep, g1_swing_foot_clearance_budget(), field,
              previous, current, radius,
              error, static_cast<int>(sizeof(error))) == G1ClearanceOk,
          error[0] == '\0' ?
              "Task 5 swept foot certifies" : error);
    check(clearance_result_same(sweep, expected_sweep),
          "Task 5 swept foot uses the supplied actual endpoint bits");

    G1ClearanceResult reversed = seeded_result(607.0);
    check(g1_swept_foot_clearance(
              reversed, g1_swing_foot_clearance_budget(), field,
              current, previous, radius,
              error, static_cast<int>(sizeof(error))) == G1ClearanceOk &&
          clearance_result_same(sweep, reversed),
          "Task 5 four-capsule sweep is reversal invariant");

    vec3 fake_current[4] = {
        current[0], current[1], current[2], current[3]
    };
    fake_current[2].y = float_from_bits(float_bits(current[2].y) + 1);
    const G1ClearanceResult fake_sweep = task5_direct_swept_foot(
        field, previous, fake_current, radius, 0);
    check(!clearance_result_same(expected_sweep, fake_sweep),
          "Task 5 sweep oracle distinguishes a one-ULP fake endpoint");

    vec3 positions[G1_BoneCount];
    quat rotations[G1_BoneCount];
    task5_make_pose(positions, rotations);
    const slice1d<vec3> position_slice(G1_BoneCount, positions);
    const slice1d<quat> rotation_slice(G1_BoneCount, rotations);

    const G1LegConfig left_config = g1_left_leg_config();
    const G1LegClearance expected_left = task5_direct_leg(
        field, position_slice, rotation_slice, left_config, 0);
    G1LegClearance left = {};
    left.minimum = seeded_result(611.0);
    check(g1_measure_leg_clearance(
              left, g1_pose_clearance_budget(), field,
              position_slice, rotation_slice, left_config,
              error, static_cast<int>(sizeof(error))) == G1ClearanceOk,
          error[0] == '\0' ?
              "Task 5 leg aggregate certifies" : error);
    check(task5_leg_same(left, expected_left),
          "Task 5 leg members equal direct transformed primitives");
    check(left.knee.witness.primitive_index == 0 &&
              left.ankle.witness.primitive_index == 1 &&
              left.toe.witness.primitive_index == 2 &&
              left.foot.witness.primitive_index >= 3 &&
              left.foot.witness.primitive_index <= 6 &&
              left.thigh.witness.primitive_index == 7 &&
              left.shin.witness.primitive_index == 8,
          "Task 5 standalone leg owns primitive indices 0 through 8");

    G1PoseClearance expected_pose = {};
    expected_pose.hips = task5_require_point(
        field, positions[G1_Hips], 0);
    expected_pose.left = task5_direct_leg(
        field, position_slice, rotation_slice,
        g1_left_leg_config(), 1);
    expected_pose.right = task5_direct_leg(
        field, position_slice, rotation_slice,
        g1_right_leg_config(), 10);
    const G1ClearanceResult pose_components[] = {
        expected_pose.hips,
        expected_pose.left.minimum,
        expected_pose.right.minimum
    };
    expected_pose.minimum = task5_aggregate_results(
        pose_components,
        static_cast<int>(
            sizeof(pose_components) / sizeof(pose_components[0])));

    G1PoseClearance pose = {};
    pose.minimum = seeded_result(613.0);
    check(g1_measure_pose_clearance(
              pose, g1_pose_clearance_budget(), field,
              position_slice, rotation_slice,
              error, static_cast<int>(sizeof(error))) == G1ClearanceOk,
          error[0] == '\0' ?
              "Task 5 pose aggregate certifies" : error);
    check(clearance_result_same(pose.hips, expected_pose.hips) &&
              task5_leg_same(pose.left, expected_pose.left) &&
              task5_leg_same(pose.right, expected_pose.right) &&
              clearance_result_same(
                  pose.minimum, expected_pose.minimum),
          "Task 5 pose members equal all nineteen direct primitives");
    check(pose.hips.witness.primitive_index == 0 &&
              pose.left.knee.witness.primitive_index == 1 &&
              pose.left.ankle.witness.primitive_index == 2 &&
              pose.left.toe.witness.primitive_index == 3 &&
              pose.left.foot.witness.primitive_index >= 4 &&
              pose.left.foot.witness.primitive_index <= 7 &&
              pose.left.thigh.witness.primitive_index == 8 &&
              pose.left.shin.witness.primitive_index == 9 &&
              pose.right.knee.witness.primitive_index == 10 &&
              pose.right.ankle.witness.primitive_index == 11 &&
              pose.right.toe.witness.primitive_index == 12 &&
              pose.right.foot.witness.primitive_index >= 13 &&
              pose.right.foot.witness.primitive_index <= 16 &&
              pose.right.thigh.witness.primitive_index == 17 &&
              pose.right.shin.witness.primitive_index == 18,
          "Task 5 pose owns Hips 0, left 1..9, and right 10..18");
    check(double_bits(pose.left.minimum.lower_bound_m) ==
              double_bits(pose.right.minimum.lower_bound_m) &&
              pose.minimum.witness.primitive_index < 10,
          "Task 5 pose minimum tie selects the stable left primitive");
}

static void task5_require_foot_failure(
    const heightfield& field,
    const vec3 centers[4],
    float radius,
    const G1ClearanceBudget& limits,
    G1ClearanceStatus expected,
    double seed,
    const char* message)
{
    G1ClearanceResult output = seeded_result(seed);
    const ByteSnapshot<G1ClearanceResult> output_before(output);
    const ByteSnapshot<G1ClearanceBudget> limits_before(limits);
    char error[128] = {};
    check(g1_foot_clearance(
              output, limits, field, centers, radius,
              error, static_cast<int>(sizeof(error))) == expected,
          message);
    check(output_before.same(output) && limits_before.same(limits),
          "Task 5 failed foot aggregate preserves output and limits");
}

static void test_task5_aggregate_transactions()
{
    heightfield field;
    point_make_field(field, 9, 9, 0.0f, 0.0f, 0.25f);
    const vec3 valid[4] = {
        vec3(0.30f, 0.80f, 0.30f),
        vec3(0.55f, 0.90f, 0.30f),
        vec3(0.30f, 1.00f, 0.55f),
        vec3(0.55f, 1.10f, 0.55f)
    };
    const float radius = 0.02f;

    vec3 outside[4] = {valid[0], valid[1], valid[2], valid[3]};
    outside[0].x = 0.0f;
    task5_require_foot_failure(
        field, outside, radius, g1_pose_clearance_budget(),
        G1ClearanceOutsideDomain, 801.0,
        "Task 5 foot reports outside-domain transactionally");

    G1ClearanceBudget no_cells = g1_pose_clearance_budget();
    no_cells.maximum_cells = 0;
    task5_require_foot_failure(
        field, valid, radius, no_cells,
        G1ClearanceBudgetExceeded, 803.0,
        "Task 5 foot reports tightened shared-budget exhaustion");

    const G1ClearanceResult direct = task5_direct_foot(
        field, valid, radius, 0);
    check(direct.work.cells_visited > 0,
          "Task 5 foot late-budget fixture consumes cells");
    G1ClearanceBudget one_cell_short = g1_pose_clearance_budget();
    one_cell_short.maximum_cells = direct.work.cells_visited - 1;
    task5_require_foot_failure(
        field, valid, radius, one_cell_short,
        G1ClearanceBudgetExceeded, 805.0,
        "Task 5 foot enforces one shared ledger across four spheres");

    heightfield wide_guard;
    point_make_field(wide_guard, 5, 5, 0.0f, 0.0f, 0.25f);
    wide_guard.heights.set(16.0f);
    const vec3 high[4] = {
        vec3(0.50f, 20.0f, 0.50f),
        vec3(0.75f, 20.0f, 0.50f),
        vec3(0.50f, 20.0f, 0.75f),
        vec3(0.75f, 20.0f, 0.75f)
    };
    task5_require_foot_failure(
        wide_guard, high, 0.125f, g1_pose_clearance_budget(),
        G1ClearanceUncertified, 807.0,
        "Task 5 foot propagates an uncertified primitive");

    vec3 invalid[4] = {valid[0], valid[1], valid[2], valid[3]};
    invalid[3].z = float_from_bits(UINT32_C(0x7fc00001));
    task5_require_foot_failure(
        field, invalid, radius, g1_pose_clearance_budget(),
        G1ClearanceInvalidInput, 809.0,
        "Task 5 foot rejects invalid input transactionally");

    heightfield invalid_field = field;
    invalid_field.version = 1;
    task5_require_foot_failure(
        invalid_field, valid, radius, g1_pose_clearance_budget(),
        G1ClearanceInvalidField, 811.0,
        "Task 5 foot rejects an invalid field transactionally");

    {
        RoundingModeGuard guard;
        check(std::fesetround(FE_UPWARD) == 0,
              "Task 5 can enter hostile rounding for aggregate failure");
        task5_require_foot_failure(
            field, valid, radius, g1_pose_clearance_budget(),
            G1ClearanceArithmeticFailure, 813.0,
            "Task 5 foot rejects hostile arithmetic transactionally");
    }

    vec3 positions[G1_BoneCount];
    quat rotations[G1_BoneCount];
    task5_make_pose(positions, rotations);
    const slice1d<vec3> position_slice(G1_BoneCount, positions);
    const slice1d<quat> rotation_slice(G1_BoneCount, rotations);

    heightfield budget_field;
    point_make_field(
        budget_field, 33, 33, -1.0f, -1.0f, 0.0625f);
    G1PoseClearance measured_pose = task5_seed_pose(815.0);
    char budget_error[256] = {};
    const G1ClearanceStatus measured_status =
        g1_measure_pose_clearance(
            measured_pose, g1_pose_clearance_budget(), budget_field,
            position_slice, rotation_slice,
            budget_error, static_cast<int>(sizeof(budget_error)));
    check(measured_status == G1ClearanceOk,
          budget_error[0] == '\0' ?
              "Task 5 late pose-budget baseline certifies" :
              budget_error);
    check(measured_pose.minimum.work.cells_visited > 19,
          "Task 5 late pose-budget fixture has nontrivial shared work");
    G1ClearanceBudget pose_one_cell_short =
        g1_pose_clearance_budget();
    pose_one_cell_short.maximum_cells =
        measured_pose.minimum.work.cells_visited - 1;
    G1PoseClearance budget_pose = task5_seed_pose(817.0);
    const ByteSnapshot<G1PoseClearance> budget_pose_before(budget_pose);
    check(g1_measure_pose_clearance(
              budget_pose, pose_one_cell_short, budget_field,
              position_slice, rotation_slice,
              NULL, 0) == G1ClearanceBudgetExceeded &&
              budget_pose_before.same(budget_pose),
          "Task 5 late shared-budget failure preserves every pose byte");

    positions[g1_left_leg_config().ankle].x =
        float_from_bits(UINT32_C(0x7fc00002));
    G1LegClearance leg = task5_seed_leg(820.0);
    const ByteSnapshot<G1LegClearance> leg_before(leg);
    check(g1_measure_leg_clearance(
              leg, g1_pose_clearance_budget(), field,
              position_slice, rotation_slice, g1_left_leg_config(),
              NULL, 0) == G1ClearanceInvalidInput &&
              leg_before.same(leg),
          "Task 5 invalid transformed leg preserves every output byte");

    task5_make_pose(positions, rotations);
    positions[g1_right_leg_config().ankle].z =
        float_from_bits(UINT32_C(0x7fc00003));
    G1PoseClearance pose = task5_seed_pose(840.0);
    const ByteSnapshot<G1PoseClearance> pose_before(pose);
    check(g1_measure_pose_clearance(
              pose, g1_pose_clearance_budget(), field,
              position_slice, rotation_slice,
              NULL, 0) == G1ClearanceInvalidInput &&
              pose_before.same(pose),
          "Task 5 late right-leg input preserves the complete pose output");
}

static int run_task5_aggregate_mode()
{
    test_task5_aggregate_certificates();
    test_task5_aggregate_transactions();
    return 0;
}

static G1SwingHistory task5_unique_history(uint32_t seed)
{
    G1SwingHistory history = {};
    history.initialized = true;
    for (uint32_t index = 0; index < 4; ++index) {
        history.previous_sphere_centers[index] = vec3(
            float_from_bits(UINT32_C(0x3e000001) + seed + index),
            float_from_bits(UINT32_C(0x3f000001) + seed + index),
            float_from_bits(UINT32_C(0x3e800001) + seed + index));
    }
    return history;
}

static G1SwingClearanceValidation task5_unique_swing_output(
    uint32_t seed)
{
    G1SwingClearanceValidation output = {};
    output.lower_margin_m = 701.0 + static_cast<double>(seed);
    output.witness_upper_m = 702.0 + static_cast<double>(seed);
    output.sweep_evaluated = true;
    output.work.point_queries = UINT32_C(0x12340000) + seed;
    output.work.cells_visited = UINT32_C(0x23450000) + seed;
    output.work.primitive_triangle_pairs = UINT32_C(0x34560000) + seed;
    output.work.face_patches = UINT32_C(0x45670000) + seed;
    output.work.candidate_tests = UINT32_C(0x56780000) + seed;
    output.work.subdivision_nodes = UINT32_C(0x67890000) + seed;
    return output;
}

static void test_task5_checked_history()
{
    const vec3 initial[4] = {
        vec3(-0.0f, 0.25f, -0.25f),
        vec3(0.10f, 0.30f, 0.20f),
        vec3(0.20f, 0.35f, 0.15f),
        vec3(0.30f, 0.40f, 0.10f)
    };
    char error[128] = {};
    G1SwingHistory history = task5_unique_history(UINT32_C(0x11));
    check(g1_swing_history_reset(
              history, initial,
              error, static_cast<int>(sizeof(error))),
          error[0] == '\0' ?
              "Task 5 history reset accepts twelve runtime components" :
              error);
    check(history.initialized &&
              float_bits(history.previous_sphere_centers[0].x) == 0 &&
              float_bits(history.previous_sphere_centers[0].z) ==
                  float_bits(-0.25f),
          "Task 5 history reset canonicalizes zero and assigns all centers");

    vec3 invalid[4] = {
        initial[0], initial[1], initial[2], initial[3]
    };
    invalid[3].y = float_from_bits(UINT32_C(0x7fc00001));
    const ByteSnapshot<G1SwingHistory> before_invalid_reset(history);
    check(!g1_swing_history_reset(
              history, invalid,
              error, static_cast<int>(sizeof(error))) &&
              before_invalid_reset.same(history),
          "Task 5 failed history reset preserves every prior byte");

    vec3 subnormal[4] = {
        initial[0], initial[1], initial[2], initial[3]
    };
    subnormal[2].z = std::numeric_limits<float>::denorm_min();
    const ByteSnapshot<G1SwingHistory> before_subnormal(history);
    check(!g1_swing_history_reset(
              history, subnormal, NULL, 0) &&
              before_subnormal.same(history),
          "Task 5 history rejects a nonzero subnormal transactionally");

    G1SwingHistory uninitialized = task5_unique_history(UINT32_C(0x22));
    uninitialized.initialized = false;
    const ByteSnapshot<G1SwingHistory> before_uninitialized(uninitialized);
    check(!g1_swing_history_commit(
              uninitialized, initial,
              error, static_cast<int>(sizeof(error))) &&
              before_uninitialized.same(uninitialized),
          "Task 5 commit requires initialized history transactionally");

    const ByteSnapshot<G1SwingHistory> before_invalid_commit(history);
    check(!g1_swing_history_commit(
              history, invalid,
              error, static_cast<int>(sizeof(error))) &&
              before_invalid_commit.same(history),
          "Task 5 failed history commit preserves every prior byte");

    const vec3 accepted[4] = {
        vec3(0.40f, 0.45f, 0.05f),
        vec3(0.50f, 0.50f, -0.0f),
        vec3(0.60f, 0.55f, -0.05f),
        vec3(0.70f, 0.60f, -0.10f)
    };
    check(g1_swing_history_commit(
              history, accepted,
              error, static_cast<int>(sizeof(error))),
          error[0] == '\0' ?
              "Task 5 initialized history accepts committed centers" :
              error);
    check(history.initialized &&
              float_bits(history.previous_sphere_centers[1].z) == 0 &&
              float_bits(history.previous_sphere_centers[3].x) ==
                  float_bits(accepted[3].x),
          "Task 5 commit assigns the complete canonical candidate");
}

static void task5_make_asymmetric_swing_field(heightfield& field)
{
    field.version = 2;
    field.nx = 9;
    field.nz = 9;
    field.origin_x = float_from_bits(UINT32_C(0x00000000));
    field.origin_z = float_from_bits(UINT32_C(0x00000000));
    field.cell_size = float_from_bits(UINT32_C(0x3e000000));
    field.exterior_height = float_from_bits(UINT32_C(0xbf800000));
    field.heights.resize(81);
    for (int z = 0; z < field.nz; ++z) {
        for (int x = 0; x < field.nx; ++x) {
            field.heights(z * field.nx + x) =
                x >= 5 && z >= 5
                    ? float_from_bits(UINT32_C(0x3b800000))
                    : float_from_bits(UINT32_C(0x00000000));
        }
    }
}

static void task5_make_exact_swing_centers(
    vec3 previous[4],
    vec3 current[4])
{
    const float previous_y =
        float_from_bits(UINT32_C(0x3c23d70a));
    const float current_y =
        float_from_bits(UINT32_C(0x3cf5c28f));
    const vec3 previous_candidate[4] = {
        vec3(float_from_bits(UINT32_C(0x3e800000)), previous_y,
             float_from_bits(UINT32_C(0x3e800000))),
        vec3(float_from_bits(UINT32_C(0x3f000000)), previous_y,
             float_from_bits(UINT32_C(0x3e800000))),
        vec3(float_from_bits(UINT32_C(0x3e800000)), previous_y,
             float_from_bits(UINT32_C(0x3f000000))),
        vec3(float_from_bits(UINT32_C(0x3f400000)), previous_y,
             float_from_bits(UINT32_C(0x3f400000)))
    };
    const vec3 current_candidate[4] = {
        vec3(float_from_bits(UINT32_C(0x3e880000)), current_y,
             float_from_bits(UINT32_C(0x3e880000))),
        vec3(float_from_bits(UINT32_C(0x3f040000)), current_y,
             float_from_bits(UINT32_C(0x3e880000))),
        vec3(float_from_bits(UINT32_C(0x3e880000)), current_y,
             float_from_bits(UINT32_C(0x3f040000))),
        vec3(float_from_bits(UINT32_C(0x3f440000)), current_y,
             float_from_bits(UINT32_C(0x3f440000)))
    };
    for (int index = 0; index < 4; ++index) {
        previous[index] = previous_candidate[index];
        current[index] = current_candidate[index];
    }
}

static void test_task5_actual_center_swing()
{
    heightfield field;
    task5_make_asymmetric_swing_field(field);
    const G1LegConfig config = g1_left_leg_config();
    vec3 previous[4] = {};
    vec3 current[4] = {};
    task5_make_exact_swing_centers(previous, current);
    check(static_cast<double>(previous[0].y) -
                  static_cast<double>(config.planted_clearance_m) ==
              static_cast<double>(config.planted_clearance_m) &&
              static_cast<double>(current[0].y) -
                  static_cast<double>(config.swing_clearance_m) ==
              static_cast<double>(config.swing_clearance_m),
          "Task 5 direct swing targets are exactly binary32 representable");
    char error[256] = {};
    G1SwingHistory history = task5_unique_history(UINT32_C(0x33));
    check(g1_swing_history_reset(
              history, previous,
              error, static_cast<int>(sizeof(error))),
          "Task 5 swing fixture initializes history");

    vec3 adjusted_previous[4] = {};
    vec3 adjusted_current[4] = {};
    for (int index = 0; index < 4; ++index) {
        adjusted_previous[index] = vec3(
            previous[index].x,
            config.planted_clearance_m,
            previous[index].z);
        adjusted_current[index] = vec3(
            current[index].x,
            config.swing_clearance_m,
            current[index].z);
    }
    const G1ClearanceResult direct = task5_direct_swept_foot(
        field, adjusted_previous, adjusted_current,
        config.foot_sphere_radius_m, 0);
    const G1ClearanceResult first = task5_require_capsule(
        field, adjusted_previous[0], adjusted_current[0],
        config.foot_sphere_radius_m, 0,
        g1_swing_foot_clearance_budget());
    check(direct.witness.primitive_index == 3 &&
              (double_bits(direct.lower_bound_m) !=
                   double_bits(first.lower_bound_m) ||
               double_bits(direct.witness_upper_m) !=
                   double_bits(first.witness_upper_m)),
          "Task 5 asymmetric swing oracle has a later-sphere winner");
    G1SwingClearanceValidation output =
        task5_unique_swing_output(UINT32_C(0x34));
    check(g1_swing_clearance_validate(
              output, g1_swing_foot_clearance_budget(),
              history, field, config, current,
              false, 0.04f,
              error, static_cast<int>(sizeof(error))) == G1ClearanceOk,
          error[0] == '\0' ?
              "Task 5 actual-center sweep certifies" : error);
    check(output.sweep_evaluated &&
              double_bits(output.lower_margin_m) ==
                  double_bits(direct.lower_bound_m) &&
              double_bits(output.witness_upper_m) ==
                  double_bits(direct.witness_upper_m) &&
              clearance_work_same(output.work, direct.work) &&
              output.witness_upper_m - output.lower_margin_m <=
                  G1ClearanceMaximumCertificateWidthM,
          "Task 5 swing exactly matches four direct target-subtracted capsules");

    G1SwingClearanceValidation contact =
        task5_unique_swing_output(UINT32_C(0x35));
    heightfield bypass_field;
    check(g1_swing_clearance_validate(
              contact, g1_swing_foot_clearance_budget(),
              history, bypass_field, config, current,
              true, 0.04f,
              error, static_cast<int>(sizeof(error))) == G1ClearanceOk &&
              !contact.sweep_evaluated &&
              double_bits(contact.lower_margin_m) == 0 &&
              double_bits(contact.witness_upper_m) == 0 &&
              clearance_work_same(contact.work, G1ClearanceWork{}),
          "Task 5 recorded contact bypass returns a zero-work success");

    const float exact_dt = 0.04f;
    const float adjacent_dt[] = {
        std::nextafter(exact_dt, 0.0f),
        std::nextafter(
            exact_dt, std::numeric_limits<float>::infinity())
    };
    for (const float dt : adjacent_dt) {
        G1SwingClearanceValidation rejected =
            task5_unique_swing_output(UINT32_C(0x36));
        const ByteSnapshot<G1SwingClearanceValidation> before(rejected);
        check(g1_swing_clearance_validate(
                  rejected, g1_swing_foot_clearance_budget(),
                  history, field, config, current,
                  true, dt, NULL, 0) == G1ClearanceInvalidInput &&
                  before.same(rejected),
              "Task 5 exact 25 Hz gate rejects each adjacent dt");
    }
}

static void test_task5_actual_endpoint_rounding_trap()
{
    heightfield field;
    point_make_field(field, 5, 5, 0.0f, 0.0f, 0.25f);
    field.heights.set(float_from_bits(UINT32_C(0x30000000)));
    const G1LegConfig config = g1_left_leg_config();
    check(float_bits(config.swing_clearance_m) ==
              UINT32_C(0x3c75c28f) &&
              float_bits(config.foot_sphere_radius_m) ==
              UINT32_C(0x3ca3d70a),
          "Task 5 rounding trap uses the locked target and radius bits");

    const vec3 previous[4] = {
        vec3(0.50f, 1.0f, 0.50f),
        vec3(0.75f, 1.0f, 0.50f),
        vec3(0.50f, 1.0f, 0.75f),
        vec3(0.75f, 1.0f, 0.75f)
    };
    vec3 actual[4] = {
        vec3(0.50f, float_from_bits(UINT32_C(0x3d0f5c29)), 0.50f),
        vec3(0.75f, 0.50f, 0.50f),
        vec3(0.50f, 0.50f, 0.75f),
        vec3(0.75f, 0.50f, 0.75f)
    };
    G1SwingHistory history = task5_unique_history(UINT32_C(0x41));
    check(g1_swing_history_reset(history, previous, NULL, 0),
          "Task 5 rounding trap initializes history");

    G1SwingClearanceValidation actual_output =
        task5_unique_swing_output(UINT32_C(0x42));
    char error[256] = {};
    check(g1_swing_clearance_validate(
              actual_output, g1_swing_foot_clearance_budget(),
              history, field, config, actual,
              false, 0.04f,
              error, static_cast<int>(sizeof(error))) == G1ClearanceOk,
          error[0] == '\0' ?
              "Task 5 exact actual endpoint certifies" : error);
    check(actual_output.sweep_evaluated &&
              actual_output.lower_margin_m >= 0.0 &&
              actual_output.witness_upper_m > 0.0,
          "Task 5 binary64 target subtraction preserves positive margin");

    float materialized = float_from_bits(UINT32_C(0x41234567));
    check(g1_apply_swing_lift_y(
              materialized,
              float_from_bits(UINT32_C(0x3d0f5c28)),
              float_from_bits(UINT32_C(0x31000000)),
              NULL, 0) == G1ClearanceOk &&
              float_bits(materialized) == UINT32_C(0x3d0f5c28) &&
              float_bits(materialized) + 1 == float_bits(actual[0].y),
          "Task 5 fake baseline-plus-lift endpoint is one ULP low");
    vec3 fake[4] = {actual[0], actual[1], actual[2], actual[3]};
    fake[0].y = materialized;
    G1SwingClearanceValidation fake_output =
        task5_unique_swing_output(UINT32_C(0x43));
    check(g1_swing_clearance_validate(
              fake_output, g1_swing_foot_clearance_budget(),
              history, field, config, fake,
              false, 0.04f,
              error, static_cast<int>(sizeof(error))) == G1ClearanceOk &&
              fake_output.witness_upper_m < 0.0,
          "Task 5 one-ULP fake endpoint is rejected by its negative margin");
}

static void task5_require_swing_failure(
    const G1SwingHistory& history,
    const heightfield& field,
    const vec3 current[4],
    const G1ClearanceBudget& limits,
    float dt,
    G1ClearanceStatus expected,
    uint32_t seed,
    const char* message)
{
    G1SwingClearanceValidation output = task5_unique_swing_output(seed);
    const ByteSnapshot<G1SwingClearanceValidation> output_before(output);
    const ByteSnapshot<G1SwingHistory> history_before(history);
    const ByteSnapshot<G1ClearanceBudget> limits_before(limits);
    check(g1_swing_clearance_validate(
              output, limits, history, field, g1_left_leg_config(),
              current, false, dt, NULL, 0) == expected,
          message);
    check(output_before.same(output) &&
              history_before.same(history) &&
              limits_before.same(limits),
          "Task 5 failed swing preserves output, history, and limits");
}

static void test_task5_swing_transactions()
{
    heightfield field;
    point_make_field(field, 5, 5, 0.0f, 0.0f, 0.25f);
    const vec3 previous[4] = {
        vec3(0.30f, 0.50f, 0.30f),
        vec3(0.55f, 0.50f, 0.30f),
        vec3(0.30f, 0.50f, 0.55f),
        vec3(0.55f, 0.50f, 0.55f)
    };
    const vec3 current[4] = {
        vec3(0.32f, 0.55f, 0.32f),
        vec3(0.57f, 0.55f, 0.32f),
        vec3(0.32f, 0.55f, 0.57f),
        vec3(0.57f, 0.55f, 0.57f)
    };
    G1SwingHistory history = task5_unique_history(UINT32_C(0x51));
    check(g1_swing_history_reset(history, previous, NULL, 0),
          "Task 5 transaction fixture initializes history");

    vec3 outside_previous[4] = {
        previous[0], previous[1], previous[2], previous[3]
    };
    vec3 outside_current[4] = {
        current[0], current[1], current[2], current[3]
    };
    outside_previous[0].x = 0.0f;
    outside_current[0].x = 0.0f;
    G1SwingHistory outside_history =
        task5_unique_history(UINT32_C(0x52));
    check(g1_swing_history_reset(
              outside_history, outside_previous, NULL, 0),
          "Task 5 outside fixture initializes history");
    task5_require_swing_failure(
        outside_history, field, outside_current,
        g1_swing_foot_clearance_budget(), 0.04f,
        G1ClearanceOutsideDomain, UINT32_C(0x53),
        "Task 5 swing reports outside-domain transactionally");

    G1ClearanceBudget no_cells = g1_swing_foot_clearance_budget();
    no_cells.maximum_cells = 0;
    task5_require_swing_failure(
        history, field, current, no_cells, 0.04f,
        G1ClearanceBudgetExceeded, UINT32_C(0x54),
        "Task 5 swing reports shared-budget exhaustion transactionally");

    heightfield wide_guard;
    point_make_field(wide_guard, 5, 5, 0.0f, 0.0f, 0.25f);
    wide_guard.heights.set(16.0f);
    const vec3 high[4] = {
        vec3(0.30f, 20.0f, 0.30f),
        vec3(0.55f, 20.0f, 0.30f),
        vec3(0.30f, 20.0f, 0.55f),
        vec3(0.55f, 20.0f, 0.55f)
    };
    G1SwingHistory high_history =
        task5_unique_history(UINT32_C(0x55));
    check(g1_swing_history_reset(high_history, high, NULL, 0),
          "Task 5 uncertified fixture initializes history");
    task5_require_swing_failure(
        high_history, wide_guard, high,
        g1_swing_foot_clearance_budget(), 0.04f,
        G1ClearanceUncertified, UINT32_C(0x56),
        "Task 5 swing propagates uncertified geometry transactionally");

    G1SwingHistory uninitialized = history;
    uninitialized.initialized = false;
    task5_require_swing_failure(
        uninitialized, field, current,
        g1_swing_foot_clearance_budget(), 0.04f,
        G1ClearanceInvalidInput, UINT32_C(0x57),
        "Task 5 swing rejects uninitialized history transactionally");

    heightfield invalid_field = field;
    invalid_field.version = 1;
    task5_require_swing_failure(
        history, invalid_field, current,
        g1_swing_foot_clearance_budget(), 0.04f,
        G1ClearanceInvalidField, UINT32_C(0x58),
        "Task 5 swing rejects an invalid field transactionally");

    {
        RoundingModeGuard guard;
        check(std::fesetround(FE_DOWNWARD) == 0,
              "Task 5 can enter hostile rounding for swing failure");
        task5_require_swing_failure(
            history, field, current,
            g1_swing_foot_clearance_budget(), 0.04f,
            G1ClearanceArithmeticFailure, UINT32_C(0x59),
            "Task 5 swing rejects hostile arithmetic transactionally");
    }
}

static int run_task5_swing_mode()
{
    test_task5_checked_history();
    test_task5_actual_center_swing();
    test_task5_actual_endpoint_rounding_trap();
    test_task5_swing_transactions();
    return 0;
}

static void task5_parity_emit_result(
    const char* name,
    G1ClearanceStatus status,
    const G1ClearanceResult& result,
    bool unchanged)
{
    const G1ClearanceWitness& witness = result.witness;
    const G1ClearanceWork& work = result.work;
    std::printf(
        "task1=%s status=%u unchanged=%u bounds=%016llx,%016llx "
        "key=%u,%d,%d,%u,%u,%u,%u work=%u,%u,%u,%u,%u,%u\n",
        name,
        static_cast<unsigned int>(status),
        unchanged ? 1U : 0U,
        static_cast<unsigned long long>(
            double_bits(result.lower_bound_m)),
        static_cast<unsigned long long>(
            double_bits(result.witness_upper_m)),
        witness.primitive_index,
        witness.cell_x,
        witness.cell_z,
        witness.terrain_triangle_index,
        witness.patch_index,
        witness.candidate_kind,
        witness.candidate_subindex,
        work.point_queries,
        work.cells_visited,
        work.primitive_triangle_pairs,
        work.face_patches,
        work.candidate_tests,
        work.subdivision_nodes);
}

static void task5_parity_emit_leg(
    const char* prefix,
    G1ClearanceStatus status,
    const G1LegClearance& leg)
{
    const G1ClearanceResult* const members[] = {
        &leg.knee, &leg.ankle, &leg.toe, &leg.foot,
        &leg.thigh, &leg.shin, &leg.minimum
    };
    const char* const suffixes[] = {
        "knee", "ankle", "toe", "foot",
        "thigh", "shin", "minimum"
    };
    for (size_t index = 0;
         index < sizeof(members) / sizeof(members[0]);
         ++index) {
        char name[64] = {};
        const int written = std::snprintf(
            name, sizeof(name), "%s-%s", prefix, suffixes[index]);
        check(written > 0 &&
                  written < static_cast<int>(sizeof(name)),
              "Task 1 parity leg record name fits");
        task5_parity_emit_result(name, status, *members[index], false);
    }
}

static void task5_parity_emit_pose(
    G1ClearanceStatus status,
    const G1PoseClearance& pose)
{
    task5_parity_emit_result(
        "pose-hips", status, pose.hips, false);
    task5_parity_emit_leg("pose-left", status, pose.left);
    task5_parity_emit_leg("pose-right", status, pose.right);
    task5_parity_emit_result(
        "pose-minimum", status, pose.minimum, false);
}

static void task5_parity_emit_history(
    const char* name,
    bool ok,
    bool unchanged,
    const G1SwingHistory& history)
{
    std::printf(
        "task1=%s ok=%u unchanged=%u initialized=%u centers=",
        name, ok ? 1U : 0U, unchanged ? 1U : 0U,
        history.initialized ? 1U : 0U);
    for (int index = 0; index < 4; ++index) {
        std::printf(
            "%s%08x,%08x,%08x",
            index == 0 ? "" : "|",
            float_bits(history.previous_sphere_centers[index].x),
            float_bits(history.previous_sphere_centers[index].y),
            float_bits(history.previous_sphere_centers[index].z));
    }
    std::printf("\n");
}

static void task5_parity_emit_swing(
    const char* name,
    G1ClearanceStatus status,
    bool unchanged,
    const G1SwingClearanceValidation& output)
{
    const G1ClearanceWork& work = output.work;
    std::printf(
        "task1=%s status=%u unchanged=%u margins=%016llx,%016llx "
        "evaluated=%u work=%u,%u,%u,%u,%u,%u\n",
        name,
        static_cast<unsigned int>(status),
        unchanged ? 1U : 0U,
        static_cast<unsigned long long>(
            double_bits(output.lower_margin_m)),
        static_cast<unsigned long long>(
            double_bits(output.witness_upper_m)),
        output.sweep_evaluated ? 1U : 0U,
        work.point_queries,
        work.cells_visited,
        work.primitive_triangle_pairs,
        work.face_patches,
        work.candidate_tests,
        work.subdivision_nodes);
}

static void task5_make_parity_field(heightfield& field)
{
    field.version = 2;
    field.nx = 17;
    field.nz = 17;
    field.origin_x = float_from_bits(UINT32_C(0xbf800000));
    field.origin_z = float_from_bits(UINT32_C(0xbf800000));
    field.cell_size = float_from_bits(UINT32_C(0x3e000000));
    field.exterior_height = float_from_bits(UINT32_C(0xbf800000));
    field.heights.resize(17 * 17);
    field.heights.set(float_from_bits(UINT32_C(0x00000000)));
}

static int run_task5_parity_mode()
{
    test_normal_arithmetic_environment();
    heightfield field;
    task5_make_parity_field(field);
    char error[256] = {};

    const vec3 foot_centers[4] = {
        vec3(float_from_bits(UINT32_C(0xbee66666)),
             float_from_bits(UINT32_C(0x3f666666)),
             float_from_bits(UINT32_C(0xbe4ccccd))),
        vec3(float_from_bits(UINT32_C(0xbe19999a)),
             float_from_bits(UINT32_C(0x3f8ccccd)),
             float_from_bits(UINT32_C(0xbdcccccd))),
        vec3(float_from_bits(UINT32_C(0x3e19999a)),
             float_from_bits(UINT32_C(0x3f4ccccd)),
             float_from_bits(UINT32_C(0x3dcccccd))),
        vec3(float_from_bits(UINT32_C(0x3ee66666)),
             float_from_bits(UINT32_C(0x3f4ccccd)),
             float_from_bits(UINT32_C(0x3e4ccccd)))
    };
    G1ClearanceResult foot = seeded_result(901.0);
    G1ClearanceStatus status = g1_foot_clearance(
        foot, g1_pose_clearance_budget(), field,
        foot_centers, float_from_bits(UINT32_C(0x3ca3d70a)),
        error, static_cast<int>(sizeof(error)));
    check(status == G1ClearanceOk,
          "Task 1 parity foot fixture certifies");
    task5_parity_emit_result("foot-ok", status, foot, false);

    const vec3 previous[4] = {
        foot_centers[0], foot_centers[1], foot_centers[2], foot_centers[3]
    };
    const vec3 current[4] = {
        vec3(float_from_bits(UINT32_C(0xbed70a3d)),
             float_from_bits(UINT32_C(0x3f68f5c3)),
             float_from_bits(UINT32_C(0xbe3851ec))),
        vec3(float_from_bits(UINT32_C(0xbdf5c28f)),
             float_from_bits(UINT32_C(0x3f8f5c29)),
             float_from_bits(UINT32_C(0xbd8f5c29))),
        vec3(float_from_bits(UINT32_C(0x3e3851ec)),
             float_from_bits(UINT32_C(0x3f4f5c29)),
             float_from_bits(UINT32_C(0x3df5c28f))),
        vec3(float_from_bits(UINT32_C(0x3ef5c28f)),
             float_from_bits(UINT32_C(0x3f4f5c29)),
             float_from_bits(UINT32_C(0x3e6147ae)))
    };
    G1ClearanceResult swept = seeded_result(903.0);
    status = g1_swept_foot_clearance(
        swept, g1_swing_foot_clearance_budget(), field,
        previous, current, float_from_bits(UINT32_C(0x3ca3d70a)),
        error, static_cast<int>(sizeof(error)));
    check(status == G1ClearanceOk,
          "Task 1 parity swept-foot fixture certifies");
    task5_parity_emit_result("swept-foot-ok", status, swept, false);

    vec3 positions[G1_BoneCount];
    quat rotations[G1_BoneCount];
    task5_make_pose(positions, rotations);
    const slice1d<vec3> position_slice(G1_BoneCount, positions);
    const slice1d<quat> rotation_slice(G1_BoneCount, rotations);
    G1LegClearance leg = task5_seed_leg(905.0);
    status = g1_measure_leg_clearance(
        leg, g1_pose_clearance_budget(), field,
        position_slice, rotation_slice, g1_left_leg_config(),
        error, static_cast<int>(sizeof(error)));
    check(status == G1ClearanceOk,
          "Task 1 parity leg fixture certifies");
    task5_parity_emit_leg("leg-ok", status, leg);

    G1PoseClearance pose = task5_seed_pose(907.0);
    status = g1_measure_pose_clearance(
        pose, g1_pose_clearance_budget(), field,
        position_slice, rotation_slice,
        error, static_cast<int>(sizeof(error)));
    check(status == G1ClearanceOk,
          "Task 1 parity pose fixture certifies");
    task5_parity_emit_pose(status, pose);

    heightfield swing_field;
    task5_make_asymmetric_swing_field(swing_field);
    vec3 swing_previous[4] = {};
    vec3 swing_current[4] = {};
    task5_make_exact_swing_centers(swing_previous, swing_current);
    G1SwingHistory history = task5_unique_history(UINT32_C(0x61));
    bool history_ok = g1_swing_history_reset(
        history, swing_previous,
        error, static_cast<int>(sizeof(error)));
    check(history_ok, "Task 1 parity history reset succeeds");
    task5_parity_emit_history(
        "history-reset-ok", history_ok, false, history);
    history_ok = g1_swing_history_commit(
        history, swing_current,
        error, static_cast<int>(sizeof(error)));
    check(history_ok, "Task 1 parity history commit succeeds");
    task5_parity_emit_history(
        "history-commit-ok", history_ok, false, history);

    vec3 invalid_history_centers[4] = {
        swing_current[0], swing_current[1],
        swing_current[2], swing_current[3]
    };
    invalid_history_centers[2].z =
        float_from_bits(UINT32_C(0x7fc00001));
    const ByteSnapshot<G1SwingHistory> history_before(history);
    history_ok = g1_swing_history_reset(
        history, invalid_history_centers, NULL, 0);
    check(!history_ok && history_before.same(history),
          "Task 1 parity history failure is transactional");
    task5_parity_emit_history(
        "history-reset-failure", history_ok, true, history);

    float lifted = float_from_bits(UINT32_C(0x41234567));
    status = g1_apply_swing_lift_y(
        lifted,
        float_from_bits(UINT32_C(0x3d0f5c28)),
        float_from_bits(UINT32_C(0x31000000)),
        error, static_cast<int>(sizeof(error)));
    check(status == G1ClearanceOk,
          "Task 1 parity lift fixture succeeds");
    std::printf(
        "task1=lift-ok status=%u unchanged=0 output=%08x\n",
        static_cast<unsigned int>(status), float_bits(lifted));
    const float lift_before = lifted;
    status = g1_apply_swing_lift_y(
        lifted,
        float_from_bits(UINT32_C(0x3d0f5c28)),
        float_from_bits(UINT32_C(0xbf800000)),
        NULL, 0);
    check(status == G1ClearanceInvalidInput &&
              float_bits(lifted) == float_bits(lift_before),
          "Task 1 parity lift failure is transactional");
    std::printf(
        "task1=lift-failure status=%u unchanged=1 output=%08x\n",
        static_cast<unsigned int>(status), float_bits(lifted));

    check(g1_swing_history_reset(history, swing_previous, NULL, 0),
          "Task 1 parity actual swing history resets");
    G1SwingClearanceValidation contact =
        task5_unique_swing_output(UINT32_C(0x62));
    heightfield bypass_field;
    status = g1_swing_clearance_validate(
        contact, g1_swing_foot_clearance_budget(),
        history, bypass_field, g1_left_leg_config(), swing_current,
        true, float_from_bits(UINT32_C(0x3d23d70a)),
        error, static_cast<int>(sizeof(error)));
    check(status == G1ClearanceOk && !contact.sweep_evaluated,
          "Task 1 parity contact bypass succeeds");
    task5_parity_emit_swing(
        "contact-bypass-ok", status, false, contact);

    G1SwingClearanceValidation swing =
        task5_unique_swing_output(UINT32_C(0x63));
    status = g1_swing_clearance_validate(
        swing, g1_swing_foot_clearance_budget(),
        history, swing_field, g1_left_leg_config(), swing_current,
        false, float_from_bits(UINT32_C(0x3d23d70a)),
        error, static_cast<int>(sizeof(error)));
    check(status == G1ClearanceOk && swing.sweep_evaluated,
          "Task 1 parity actual swing succeeds");
    task5_parity_emit_swing("actual-swing-ok", status, false, swing);

    heightfield invalid_field = field;
    invalid_field.version = 1;
    G1ClearanceResult failed_foot = seeded_result(911.0);
    const ByteSnapshot<G1ClearanceResult> failed_foot_before(failed_foot);
    status = g1_foot_clearance(
        failed_foot, g1_pose_clearance_budget(), invalid_field,
        foot_centers, float_from_bits(UINT32_C(0x3ca3d70a)), NULL, 0);
    check(status == G1ClearanceInvalidField &&
              failed_foot_before.same(failed_foot),
          "Task 1 parity foot failure is transactional");
    task5_parity_emit_result(
        "foot-failure", status, failed_foot, true);

    G1ClearanceBudget no_cells = g1_swing_foot_clearance_budget();
    no_cells.maximum_cells = 0;
    G1ClearanceResult failed_swept = seeded_result(913.0);
    const ByteSnapshot<G1ClearanceResult> failed_swept_before(
        failed_swept);
    status = g1_swept_foot_clearance(
        failed_swept, no_cells, field, previous, current,
        float_from_bits(UINT32_C(0x3ca3d70a)), NULL, 0);
    check(status == G1ClearanceBudgetExceeded &&
              failed_swept_before.same(failed_swept),
          "Task 1 parity swept failure is transactional");
    task5_parity_emit_result(
        "swept-foot-failure", status, failed_swept, true);

    vec3 invalid_positions[G1_BoneCount];
    quat invalid_rotations[G1_BoneCount];
    task5_make_pose(invalid_positions, invalid_rotations);
    invalid_positions[G1_LeftAnkle].x =
        float_from_bits(UINT32_C(0x7fc00002));
    G1LegClearance failed_leg = task5_seed_leg(915.0);
    const ByteSnapshot<G1LegClearance> failed_leg_before(failed_leg);
    status = g1_measure_leg_clearance(
        failed_leg, g1_pose_clearance_budget(), field,
        slice1d<vec3>(G1_BoneCount, invalid_positions),
        slice1d<quat>(G1_BoneCount, invalid_rotations),
        g1_left_leg_config(), NULL, 0);
    check(status == G1ClearanceInvalidInput &&
              failed_leg_before.same(failed_leg),
          "Task 1 parity leg failure is transactional");
    task5_parity_emit_result(
        "leg-failure", status, failed_leg.minimum, true);

    G1PoseClearance failed_pose = task5_seed_pose(917.0);
    const ByteSnapshot<G1PoseClearance> failed_pose_before(failed_pose);
    status = g1_measure_pose_clearance(
        failed_pose, g1_pose_clearance_budget(), invalid_field,
        position_slice, rotation_slice, NULL, 0);
    check(status == G1ClearanceInvalidField &&
              failed_pose_before.same(failed_pose),
          "Task 1 parity pose failure is transactional");
    task5_parity_emit_result(
        "pose-failure", status, failed_pose.minimum, true);

    G1SwingClearanceValidation failed_swing =
        task5_unique_swing_output(UINT32_C(0x64));
    const ByteSnapshot<G1SwingClearanceValidation> failed_swing_before(
        failed_swing);
    status = g1_swing_clearance_validate(
        failed_swing, g1_swing_foot_clearance_budget(),
        history, swing_field, g1_left_leg_config(), swing_current,
        false, float_from_bits(UINT32_C(0x3d23d709)), NULL, 0);
    check(status == G1ClearanceInvalidInput &&
              failed_swing_before.same(failed_swing),
          "Task 1 parity swing failure is transactional");
    task5_parity_emit_swing(
        "actual-swing-failure", status, true, failed_swing);
    return 0;
}

static void task34_parity_emit(
    const char* name,
    const heightfield& field,
    vec3 endpoint_a,
    vec3 endpoint_b,
    float radius,
    bool require_subdivision)
{
    G1ClearanceResult result = seeded_result(997.0);
    const G1ClearanceStatus status = g1_capsule_clearance(
        result, g1_pose_clearance_budget(), field,
        endpoint_a, endpoint_b, radius, NULL, 0);
    check(status == G1ClearanceOk &&
              result.lower_bound_m <= result.witness_upper_m &&
              result.witness_upper_m - result.lower_bound_m <=
                  G1ClearanceMaximumCertificateWidthM,
          "capsule parity fixture returns a certified interval");
    check((result.work.subdivision_nodes != 0) == require_subdivision,
          "capsule parity fixture has its locked fallback class");
    const G1ClearanceWitness& witness = result.witness;
    const G1ClearanceWork& work = result.work;
    std::printf(
        "clearance=%s status=%u bounds=%016llx,%016llx "
        "witness=%016llx,%016llx,%016llx,%016llx,%016llx,%016llx,"
        "%016llx,%016llx,%016llx,%016llx "
        "key=%u,%d,%d,%u,%u,%u,%u "
        "work=%u,%u,%u,%u,%u,%u\n",
        name, static_cast<unsigned int>(status),
        static_cast<unsigned long long>(
            double_bits(result.lower_bound_m)),
        static_cast<unsigned long long>(
            double_bits(result.witness_upper_m)),
        static_cast<unsigned long long>(double_bits(witness.body_x)),
        static_cast<unsigned long long>(double_bits(witness.body_y)),
        static_cast<unsigned long long>(double_bits(witness.body_z)),
        static_cast<unsigned long long>(double_bits(witness.surface_x)),
        static_cast<unsigned long long>(double_bits(witness.surface_y)),
        static_cast<unsigned long long>(double_bits(witness.surface_z)),
        static_cast<unsigned long long>(
            double_bits(witness.segment_parameter)),
        static_cast<unsigned long long>(
            double_bits(witness.terrain_weight_0)),
        static_cast<unsigned long long>(
            double_bits(witness.terrain_weight_1)),
        static_cast<unsigned long long>(
            double_bits(witness.terrain_weight_2)),
        witness.primitive_index,
        witness.cell_x,
        witness.cell_z,
        witness.terrain_triangle_index,
        witness.patch_index,
        witness.candidate_kind,
        witness.candidate_subindex,
        work.point_queries,
        work.cells_visited,
        work.primitive_triangle_pairs,
        work.face_patches,
        work.candidate_tests,
        work.subdivision_nodes);
}

static int run_task34_combined_parity_mode()
{
    test_normal_arithmetic_environment();
    const int query_status = run_query_parity_mode();
    if (query_status != 0) {
        return query_status;
    }

    heightfield analytic;
    point_make_field(analytic, 5, 5, 0.0f, 0.0f, 0.25f);
    task34_parity_emit(
        "analytic", analytic,
        vec3(float_from_bits(UINT32_C(0x3f000000)),
             float_from_bits(UINT32_C(0x3f800000)),
             float_from_bits(UINT32_C(0x3f000000))),
        vec3(float_from_bits(UINT32_C(0x3f000000)),
             float_from_bits(UINT32_C(0x3fa00000)),
             float_from_bits(UINT32_C(0x3f000000))),
        float_from_bits(UINT32_C(0x3e000000)), false);

    heightfield fallback;
    point_make_field(fallback, 3, 3, 0.0f, 0.0f, 0.25f);
    const uint32_t height_bits[] = {
        UINT32_C(0x3ec00000), UINT32_C(0x3e000000),
        UINT32_C(0x3fe00000), UINT32_C(0x3fa00000),
        UINT32_C(0x3e800000), UINT32_C(0x3e000000),
        UINT32_C(0x3fb00000), UINT32_C(0x3f900000),
        UINT32_C(0x3fe00000)
    };
    for (int index = 0; index < 9; ++index) {
        fallback.heights(index) = float_from_bits(height_bits[index]);
    }
    task34_parity_emit(
        "fallback", fallback,
        vec3(float_from_bits(UINT32_C(0x3ed00000)),
             float_from_bits(UINT32_C(0x40100000)),
             float_from_bits(UINT32_C(0x3e620000))),
        vec3(float_from_bits(UINT32_C(0x3ec60000)),
             float_from_bits(UINT32_C(0x3fc80000)),
             float_from_bits(UINT32_C(0x3e4e0000))),
        float_from_bits(UINT32_C(0x3dc00000)), true);

    heightfield kind2;
    vec3 kind2_a;
    vec3 kind2_b;
    float kind2_radius = 0.0f;
    task34_make_kind2_winner_fixture(
        kind2, kind2_a, kind2_b, kind2_radius);
    task34_parity_emit(
        "kind2", kind2,
        kind2_a, kind2_b, kind2_radius, true);
    return 0;
}

int main(int argc, char** argv)
{
    if (argc == 2 &&
        std::strcmp(argv[1], "--task5-aggregate") == 0) {
        return run_task5_aggregate_mode();
    }
    if (argc == 2 &&
        std::strcmp(argv[1], "--task5-swing") == 0) {
        return run_task5_swing_mode();
    }
    if (argc == 2 &&
        std::strcmp(argv[1], "--task5-parity") == 0) {
        return run_task5_parity_mode();
    }
    if (argc == 2 && std::strcmp(argv[1], "--query-parity") == 0) {
        return run_task34_combined_parity_mode();
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
    if (argc == 2 &&
        std::strcmp(argv[1], "--task34-geometry") == 0) {
        test_task34_fixed_diagonal_and_rank_cases();
        test_task34_membership_uncertainty_and_witness_enclosure();
        test_task34_exact_zero_objective_tie_characterization();
        return 0;
    }
    if (argc == 2 &&
        std::strcmp(argv[1], "--task34-domain-field") == 0) {
        test_task34_input_field_and_exterior_matrix();
        test_task34_domain_and_malformed_height_matrix();
        return 0;
    }
    if (argc == 2 &&
        std::strcmp(argv[1], "--task34-counts") == 0) {
        test_task34_rectangular_count_preflights();
        return 0;
    }
    if (argc == 2 &&
        std::strcmp(argv[1], "--task34-tent") == 0) {
        test_task34_named_tent_sign_reversal();
        return 0;
    }
    if (argc == 2 &&
        std::strcmp(argv[1], "--task34-guards") == 0) {
        test_task34_sphere_capsule_output_guards();
        return 0;
    }
    if (argc == 2 &&
        std::strcmp(argv[1], "--task34-fallback") == 0) {
        test_task34_exact_dyadic_fallback();
        test_task34_public_kind2_winner();
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
    test_task34_fixed_diagonal_and_rank_cases();
    test_task34_membership_uncertainty_and_witness_enclosure();
    test_task34_exact_zero_objective_tie_characterization();
    test_task34_input_field_and_exterior_matrix();
    test_task34_domain_and_malformed_height_matrix();
    test_task34_rectangular_count_preflights();
    test_task34_named_tent_sign_reversal();
    test_task34_sphere_capsule_output_guards();
    test_task34_exact_dyadic_fallback();
    test_task34_public_kind2_winner();
    test_task5_aggregate_certificates();
    test_task5_aggregate_transactions();
    test_task5_checked_history();
    test_task5_actual_center_swing();
    test_task5_actual_endpoint_rounding_trap();
    test_task5_swing_transactions();
    test_arithmetic_environment_rejection_and_restoration();
    return 0;
}
