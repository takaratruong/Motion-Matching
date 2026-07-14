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
    return g1_point_clearance(
        output, limits, field, vec3(), error, error_capacity);
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

int main()
{
    test_normal_arithmetic_environment();
    test_status_and_factory_contract();
    test_budget_contract();
    test_swing_lift_materializer();
    test_arithmetic_environment_rejection_and_restoration();
    return 0;
}
