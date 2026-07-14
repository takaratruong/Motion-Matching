#ifdef __FAST_MATH__
#error "g1_clearance.cpp certified kernel must be compiled without fast math"
#endif

#define G1_CLEARANCE_IMPLEMENTATION_TU
#include "g1_clearance.h"
#undef G1_CLEARANCE_IMPLEMENTATION_TU

#include <cfenv>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>

#if defined(__SSE__) || defined(_M_X64) || defined(_M_IX86_FP)
#include <xmmintrin.h>
#endif

static_assert(sizeof(float) == 4,
              "G1 clearance requires binary32");
static_assert(sizeof(double) == 8,
              "G1 clearance requires binary64");
static_assert(FLT_RADIX == 2 && FLT_MANT_DIG == 24 && FLT_MAX_EXP == 128,
              "G1 clearance requires the binary32 format");
static_assert(DBL_MANT_DIG == 53 && DBL_MAX_EXP == 1024,
              "G1 clearance requires the binary64 format");
static_assert(std::numeric_limits<float>::is_iec559,
              "G1 clearance requires IEEE-754 float");
static_assert(std::numeric_limits<double>::is_iec559,
              "G1 clearance requires IEEE-754 double");
static_assert(
    std::numeric_limits<float>::has_denorm == std::denorm_present,
    "G1 clearance requires binary32 gradual underflow");
static_assert(
    std::numeric_limits<double>::has_denorm == std::denorm_present,
    "G1 clearance requires binary64 gradual underflow");

#if defined(__GNUC__) || defined(__clang__)
#define G1_CLEARANCE_NOINLINE __attribute__((noinline))
#elif defined(_MSC_VER)
#define G1_CLEARANCE_NOINLINE __declspec(noinline)
#else
#define G1_CLEARANCE_NOINLINE
#endif

struct G1ClearanceProtectedRange
{
    const void* data;
    size_t size;
};

struct G1ClearanceDiagnostic
{
    char* output;
    int capacity;
};

static bool g1_clearance_address_range(
    const void* data,
    size_t size,
    uintptr_t& begin,
    uintptr_t& end)
{
    begin = reinterpret_cast<uintptr_t>(data);
    if (size > UINTPTR_MAX - begin) {
        return false;
    }
    end = begin + size;
    return true;
}

static G1ClearanceDiagnostic g1_clearance_prepare_diagnostic(
    char* output,
    int capacity,
    const G1ClearanceProtectedRange* protected_ranges,
    size_t protected_range_count)
{
    G1ClearanceDiagnostic diagnostic = {output, capacity};
    if (output == NULL || capacity <= 0) {
        return diagnostic;
    }

    uintptr_t diagnostic_begin = 0;
    uintptr_t diagnostic_end = 0;
    if (!g1_clearance_address_range(
            output, static_cast<size_t>(capacity),
            diagnostic_begin, diagnostic_end)) {
        return {NULL, 0};
    }

    for (size_t index = 0; index < protected_range_count; ++index) {
        const G1ClearanceProtectedRange& protected_range =
            protected_ranges[index];
        if (protected_range.data == NULL || protected_range.size == 0) {
            continue;
        }
        uintptr_t protected_begin = 0;
        uintptr_t protected_end = 0;
        if (!g1_clearance_address_range(
                protected_range.data, protected_range.size,
                protected_begin, protected_end)) {
            return {NULL, 0};
        }
        if (diagnostic_begin < protected_end &&
            protected_begin < diagnostic_end) {
            return {NULL, 0};
        }
    }
    return diagnostic;
}

static G1ClearanceStatus g1_clearance_error(
    G1ClearanceStatus status,
    char* output,
    int capacity,
    const char* format,
    ...)
{
    if (output != NULL && capacity > 0) {
        va_list arguments;
        va_start(arguments, format);
        std::vsnprintf(
            output, static_cast<size_t>(capacity), format, arguments);
        va_end(arguments);
    }
    return status;
}

G1_CLEARANCE_NOINLINE static bool
g1_clearance_binary32_gradual_underflow_probe()
{
    volatile float minimum = std::numeric_limits<float>::min();
    volatile float half = 0.5f;
    volatile float denormal = std::numeric_limits<float>::denorm_min();
    volatile float product = minimum * half;
    volatile float sum = denormal + denormal;
    const float materialized_product = product;
    const float materialized_sum = sum;
    uint32_t product_bits = 0;
    uint32_t sum_bits = 0;
    std::memcpy(
        &product_bits, &materialized_product, sizeof(product_bits));
    std::memcpy(&sum_bits, &materialized_sum, sizeof(sum_bits));
    return product_bits == UINT32_C(0x00400000) &&
           sum_bits == UINT32_C(0x00000002);
}

G1_CLEARANCE_NOINLINE static bool
g1_clearance_binary64_gradual_underflow_probe()
{
    volatile double minimum = std::numeric_limits<double>::min();
    volatile double half = 0.5;
    volatile double denormal = std::numeric_limits<double>::denorm_min();
    volatile double product = minimum * half;
    volatile double sum = denormal + denormal;
    const double materialized_product = product;
    const double materialized_sum = sum;
    uint64_t product_bits = 0;
    uint64_t sum_bits = 0;
    std::memcpy(
        &product_bits, &materialized_product, sizeof(product_bits));
    std::memcpy(&sum_bits, &materialized_sum, sizeof(sum_bits));
    return product_bits == UINT64_C(0x0008000000000000) &&
           sum_bits == UINT64_C(0x0000000000000002);
}

G1_CLEARANCE_NOINLINE bool
g1_clearance_arithmetic_environment_is_supported()
{
    if (std::fegetround() != FE_TONEAREST) {
        return false;
    }
#if defined(__SSE__) || defined(_M_X64) || defined(_M_IX86_FP)
    const uint32_t mxcsr = _mm_getcsr();
    if ((mxcsr & (UINT32_C(1) << 15)) != 0 ||
        (mxcsr & (UINT32_C(1) << 6)) != 0) {
        return false;
    }
#endif

    fenv_t saved_environment;
    if (std::fegetenv(&saved_environment) != 0) {
        return false;
    }

    const bool binary32_supported =
        g1_clearance_binary32_gradual_underflow_probe();
    const bool binary64_supported =
        g1_clearance_binary64_gradual_underflow_probe();
    const bool environment_restored =
        std::fesetenv(&saved_environment) == 0;
#if defined(__SSE__) || defined(_M_X64) || defined(_M_IX86_FP)
    _mm_setcsr(mxcsr);
#endif
    return binary32_supported && binary64_supported &&
           environment_restored;
}

const char* g1_clearance_status_name(G1ClearanceStatus status)
{
    switch (status) {
    case G1ClearanceOk: return "ok";
    case G1ClearanceOutsideDomain: return "outside-domain";
    case G1ClearanceBudgetExceeded: return "budget-exceeded";
    case G1ClearanceUncertified: return "uncertified";
    case G1ClearanceInvalidInput: return "invalid-input";
    case G1ClearanceInvalidField: return "invalid-field";
    case G1ClearanceArithmeticFailure: return "arithmetic-failure";
    default: return "unknown";
    }
}

G1ClearanceBudget g1_pose_clearance_budget()
{
    G1ClearanceBudget budget = {};
    budget.maximum_point_queries =
        G1ClearanceMaximumPointQueriesPerPose;
    budget.maximum_cells = G1ClearanceMaximumCellsPerPose;
    budget.maximum_primitive_triangle_pairs =
        G1ClearanceMaximumPairsPerPose;
    budget.maximum_face_patches =
        G1ClearanceMaximumPairsPerPose * G1ClearancePatchesPerPair;
    budget.maximum_candidate_tests =
        G1ClearanceMaximumPairsPerPose * G1ClearanceCandidatesPerPair;
    budget.maximum_subdivision_nodes =
        G1ClearanceMaximumSubdivisionNodes;
    return budget;
}

G1ClearanceBudget g1_swing_foot_clearance_budget()
{
    G1ClearanceBudget budget = {};
    budget.maximum_point_queries = 0;
    budget.maximum_cells = G1ClearanceMaximumCellsPerSwingFoot;
    budget.maximum_primitive_triangle_pairs =
        G1ClearanceMaximumPairsPerSwingFoot;
    budget.maximum_face_patches =
        G1ClearanceMaximumPairsPerSwingFoot * G1ClearancePatchesPerPair;
    budget.maximum_candidate_tests =
        G1ClearanceMaximumPairsPerSwingFoot *
        G1ClearanceCandidatesPerPair;
    budget.maximum_subdivision_nodes =
        G1ClearanceMaximumSubdivisionNodes;
    return budget;
}

static bool g1_clearance_budget_within(
    const G1ClearanceBudget& requested,
    const G1ClearanceBudget& ceiling)
{
    return requested.maximum_point_queries <=
               ceiling.maximum_point_queries &&
           requested.maximum_cells <= ceiling.maximum_cells &&
           requested.maximum_primitive_triangle_pairs <=
               ceiling.maximum_primitive_triangle_pairs &&
           requested.maximum_face_patches <=
               ceiling.maximum_face_patches &&
           requested.maximum_candidate_tests <=
               ceiling.maximum_candidate_tests &&
           requested.maximum_subdivision_nodes <=
               ceiling.maximum_subdivision_nodes;
}

static G1ClearanceStatus g1_clearance_contract_stub(
    void* public_output,
    size_t public_output_size,
    const G1ClearanceBudget& limits,
    const void* additional_protected_input,
    size_t additional_protected_input_size,
    bool swing_family,
    char* error,
    int error_capacity)
{
    const G1ClearanceProtectedRange protected_ranges[] = {
        {public_output, public_output_size},
        {&limits, sizeof(limits)},
        {additional_protected_input, additional_protected_input_size}
    };
    const G1ClearanceDiagnostic diagnostic =
        g1_clearance_prepare_diagnostic(
            error, error_capacity, protected_ranges,
            sizeof(protected_ranges) / sizeof(protected_ranges[0]));
    if (!g1_clearance_arithmetic_environment_is_supported()) {
        return g1_clearance_error(
            G1ClearanceArithmeticFailure,
            diagnostic.output, diagnostic.capacity,
            "G1 clearance arithmetic environment is unsupported");
    }
    const G1ClearanceBudget ceiling = swing_family
        ? g1_swing_foot_clearance_budget()
        : g1_pose_clearance_budget();
    if (!g1_clearance_budget_within(limits, ceiling)) {
        return g1_clearance_error(
            G1ClearanceInvalidInput,
            diagnostic.output, diagnostic.capacity,
            "G1 clearance budget exceeds its immutable factory ceiling");
    }
    return g1_clearance_error(
        G1ClearanceUncertified,
        diagnostic.output, diagnostic.capacity,
        "G1 certified geometry is not implemented by contract Task 1");
}

G1ClearanceStatus g1_apply_swing_lift_y(
    float& output_y,
    float input_y,
    float lift_m,
    char* error,
    int error_capacity)
{
    const G1ClearanceProtectedRange protected_ranges[] = {
        {&output_y, sizeof(output_y)}
    };
    const G1ClearanceDiagnostic diagnostic =
        g1_clearance_prepare_diagnostic(
            error, error_capacity, protected_ranges,
            sizeof(protected_ranges) / sizeof(protected_ranges[0]));
    if (!g1_clearance_arithmetic_environment_is_supported()) {
        return g1_clearance_error(
            G1ClearanceArithmeticFailure,
            diagnostic.output, diagnostic.capacity,
            "G1 clearance arithmetic environment is unsupported");
    }
    if (!terrain_float_is_normal_or_zero_query(input_y)) {
        return g1_clearance_error(
            G1ClearanceInvalidInput,
            diagnostic.output, diagnostic.capacity,
            "G1 swing sole input Y is invalid");
    }
    const uint32_t lift_bits = terrain_float_bits(lift_m);
    const uint32_t lift_magnitude =
        lift_bits & UINT32_C(0x7fffffff);
    const uint32_t lift_exponent =
        lift_magnitude & UINT32_C(0x7f800000);
    const bool lift_is_finite =
        lift_exponent != UINT32_C(0x7f800000);
    const bool lift_is_negative =
        (lift_bits & UINT32_C(0x80000000)) != 0 &&
        lift_magnitude != 0;
    if (!lift_is_finite || lift_is_negative || lift_m > 0.08f) {
        return g1_clearance_error(
            G1ClearanceInvalidInput,
            diagnostic.output, diagnostic.capacity,
            "G1 swing lift must be finite and in [0, 0.08] m");
    }

    const float canonical_input =
        (terrain_float_bits(input_y) & UINT32_C(0x7fffffff)) == 0
            ? 0.0f
            : input_y;
    const float canonical_lift = lift_magnitude == 0 ? 0.0f : lift_m;
    const volatile double sum =
        static_cast<double>(canonical_input) +
        static_cast<double>(canonical_lift);
    if (!terrain_double_is_finite(sum)) {
        return g1_clearance_error(
            G1ClearanceArithmeticFailure,
            diagnostic.output, diagnostic.capacity,
            "G1 swing sole command overflowed binary64");
    }
    const volatile float rounded = static_cast<float>(sum);
    const uint32_t rounded_bits = terrain_float_bits(rounded);
    const uint32_t rounded_magnitude =
        rounded_bits & UINT32_C(0x7fffffff);
    const uint32_t rounded_exponent =
        rounded_magnitude & UINT32_C(0x7f800000);
    if (rounded_exponent == UINT32_C(0x7f800000) ||
        (rounded_magnitude != 0 && rounded_exponent == 0)) {
        return g1_clearance_error(
            G1ClearanceArithmeticFailure,
            diagnostic.output, diagnostic.capacity,
            "G1 swing sole command is nonfinite or subnormal");
    }
    const float candidate = rounded_magnitude == 0 ? 0.0f : rounded;
    output_y = candidate;
    return G1ClearanceOk;
}

G1ClearanceStatus g1_point_clearance(
    G1ClearanceResult& output,
    const G1ClearanceBudget& limits,
    const heightfield&,
    vec3,
    char* error,
    int error_capacity)
{
    return g1_clearance_contract_stub(
        &output, sizeof(output), limits, NULL, 0,
        false, error, error_capacity);
}

G1ClearanceStatus g1_sphere_clearance(
    G1ClearanceResult& output,
    const G1ClearanceBudget& limits,
    const heightfield&,
    vec3,
    float,
    char* error,
    int error_capacity)
{
    return g1_clearance_contract_stub(
        &output, sizeof(output), limits, NULL, 0,
        false, error, error_capacity);
}

G1ClearanceStatus g1_capsule_clearance(
    G1ClearanceResult& output,
    const G1ClearanceBudget& limits,
    const heightfield&,
    vec3,
    vec3,
    float,
    char* error,
    int error_capacity)
{
    return g1_clearance_contract_stub(
        &output, sizeof(output), limits, NULL, 0,
        false, error, error_capacity);
}

G1ClearanceStatus g1_foot_clearance(
    G1ClearanceResult& output,
    const G1ClearanceBudget& limits,
    const heightfield&,
    const vec3[4],
    float,
    char* error,
    int error_capacity)
{
    return g1_clearance_contract_stub(
        &output, sizeof(output), limits, NULL, 0,
        false, error, error_capacity);
}

G1ClearanceStatus g1_swept_foot_clearance(
    G1ClearanceResult& output,
    const G1ClearanceBudget& limits,
    const heightfield&,
    const vec3[4],
    const vec3[4],
    float,
    char* error,
    int error_capacity)
{
    return g1_clearance_contract_stub(
        &output, sizeof(output), limits, NULL, 0,
        true, error, error_capacity);
}

G1ClearanceStatus g1_measure_leg_clearance(
    G1LegClearance& output,
    const G1ClearanceBudget& limits,
    const heightfield&,
    const slice1d<vec3>,
    const slice1d<quat>,
    const G1LegConfig&,
    char* error,
    int error_capacity)
{
    return g1_clearance_contract_stub(
        &output, sizeof(output), limits, NULL, 0,
        false, error, error_capacity);
}

G1ClearanceStatus g1_measure_pose_clearance(
    G1PoseClearance& output,
    const G1ClearanceBudget& limits,
    const heightfield&,
    const slice1d<vec3>,
    const slice1d<quat>,
    char* error,
    int error_capacity)
{
    return g1_clearance_contract_stub(
        &output, sizeof(output), limits, NULL, 0,
        false, error, error_capacity);
}

G1ClearanceStatus g1_swing_clearance_validate(
    G1SwingClearanceValidation& output,
    const G1ClearanceBudget& limits,
    const G1SwingHistory& history,
    const heightfield&,
    const G1LegConfig&,
    const vec3[4],
    bool,
    float,
    char* error,
    int error_capacity)
{
    return g1_clearance_contract_stub(
        &output, sizeof(output), limits, &history, sizeof(history),
        true, error, error_capacity);
}
