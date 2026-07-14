#ifdef __FAST_MATH__
#error "g1_clearance.cpp certified kernel must be compiled without fast math"
#endif

#define G1_CLEARANCE_IMPLEMENTATION_TU
#include "g1_clearance.h"
#undef G1_CLEARANCE_IMPLEMENTATION_TU

#include "g1_surface_query.h"

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

struct G1ClearanceInterval
{
    double lower;
    double upper;
};

struct G1PointTriangle
{
    double x[3];
    double z[3];
    double height[3];
    uint32_t index;
};

struct G1PointFraction
{
    double central;
    G1ClearanceInterval enclosure;
};

struct G1PointGeometry
{
    double x0;
    double x1;
    double z0;
    double z1;
    G1PointFraction x;
    G1PointFraction z;
    uint32_t triangle_index;
};

enum G1PointGeometryStatus
{
    G1PointGeometryValid,
    G1PointGeometryUncertified,
    G1PointGeometryArithmeticFailure
};

enum G1ClearanceDomainStatus
{
    G1ClearanceDomainInside,
    G1ClearanceDomainOutside,
    G1ClearanceDomainArithmeticFailure
};

static bool g1_clearance_outward_lower(
    double value,
    double& output)
{
    if (!terrain_double_is_finite(value)) {
        return false;
    }
    const volatile double next = std::nextafter(
        value, -std::numeric_limits<double>::infinity());
    if (!terrain_double_is_finite(next)) {
        return false;
    }
    output = next;
    return true;
}

static bool g1_clearance_outward_upper(
    double value,
    double& output)
{
    if (!terrain_double_is_finite(value)) {
        return false;
    }
    const volatile double next = std::nextafter(
        value, std::numeric_limits<double>::infinity());
    if (!terrain_double_is_finite(next)) {
        return false;
    }
    output = next;
    return true;
}

static bool g1_clearance_interval_add(
    const G1ClearanceInterval& left,
    const G1ClearanceInterval& right,
    G1ClearanceInterval& output)
{
    const volatile double lower = left.lower + right.lower;
    const volatile double upper = left.upper + right.upper;
    return g1_clearance_outward_lower(lower, output.lower) &&
           g1_clearance_outward_upper(upper, output.upper) &&
           output.lower <= output.upper;
}

static bool g1_clearance_interval_subtract(
    const G1ClearanceInterval& left,
    const G1ClearanceInterval& right,
    G1ClearanceInterval& output)
{
    const volatile double lower = left.lower - right.upper;
    const volatile double upper = left.upper - right.lower;
    return g1_clearance_outward_lower(lower, output.lower) &&
           g1_clearance_outward_upper(upper, output.upper) &&
           output.lower <= output.upper;
}

static bool g1_clearance_interval_multiply(
    const G1ClearanceInterval& left,
    const G1ClearanceInterval& right,
    G1ClearanceInterval& output)
{
    const volatile double products[4] = {
        left.lower * right.lower,
        left.lower * right.upper,
        left.upper * right.lower,
        left.upper * right.upper
    };
    double minimum = products[0];
    double maximum = products[0];
    for (int index = 0; index < 4; ++index) {
        if (!terrain_double_is_finite(products[index])) {
            return false;
        }
        minimum = products[index] < minimum
            ? products[index]
            : minimum;
        maximum = products[index] > maximum
            ? products[index]
            : maximum;
    }
    return g1_clearance_outward_lower(minimum, output.lower) &&
           g1_clearance_outward_upper(maximum, output.upper) &&
           output.lower <= output.upper;
}

static bool g1_clearance_interval_divide(
    const G1ClearanceInterval& numerator,
    const G1ClearanceInterval& denominator,
    G1ClearanceInterval& output)
{
    if (denominator.lower <= 0.0 && denominator.upper >= 0.0) {
        return false;
    }
    const volatile double quotients[4] = {
        numerator.lower / denominator.lower,
        numerator.lower / denominator.upper,
        numerator.upper / denominator.lower,
        numerator.upper / denominator.upper
    };
    double minimum = quotients[0];
    double maximum = quotients[0];
    for (int index = 0; index < 4; ++index) {
        if (!terrain_double_is_finite(quotients[index])) {
            return false;
        }
        minimum = quotients[index] < minimum
            ? quotients[index]
            : minimum;
        maximum = quotients[index] > maximum
            ? quotients[index]
            : maximum;
    }
    return g1_clearance_outward_lower(minimum, output.lower) &&
           g1_clearance_outward_upper(maximum, output.upper) &&
           output.lower <= output.upper;
}

static G1ClearanceDomainStatus g1_clearance_domain_contains(
    const heightfield& field,
    vec3 point)
{
    const volatile double maximum_x_product =
        static_cast<double>(field.nx - 1) *
        static_cast<double>(field.cell_size);
    const volatile double maximum_z_product =
        static_cast<double>(field.nz - 1) *
        static_cast<double>(field.cell_size);
    const volatile double maximum_x =
        static_cast<double>(field.origin_x) + maximum_x_product;
    const volatile double maximum_z =
        static_cast<double>(field.origin_z) + maximum_z_product;
    if (!terrain_double_is_finite(maximum_x_product) ||
        !terrain_double_is_finite(maximum_z_product) ||
        !terrain_double_is_finite(maximum_x) ||
        !terrain_double_is_finite(maximum_z)) {
        return G1ClearanceDomainArithmeticFailure;
    }
    const double x = static_cast<double>(point.x);
    const double z = static_cast<double>(point.z);
    if (x < static_cast<double>(field.origin_x) || x > maximum_x ||
        z < static_cast<double>(field.origin_z) || z > maximum_z) {
        return G1ClearanceDomainOutside;
    }
    return G1ClearanceDomainInside;
}

static bool g1_clearance_point_node(
    float origin,
    int index,
    float cell_size,
    double& output)
{
    if (index < 0) {
        return false;
    }
    const volatile double product =
        static_cast<double>(index) *
        static_cast<double>(cell_size);
    const volatile double node =
        static_cast<double>(origin) + product;
    if (!terrain_double_is_finite(product) ||
        !terrain_double_is_finite(node)) {
        return false;
    }
    output = node;
    return true;
}

static bool g1_clearance_point_fraction_enclosure(
    double coordinate,
    double first_node,
    double second_node,
    G1PointFraction& output)
{
    if (!terrain_double_is_finite(coordinate) ||
        !terrain_double_is_finite(first_node) ||
        !terrain_double_is_finite(second_node)) {
        return false;
    }

    const volatile double central_numerator =
        coordinate - first_node;
    const volatile double central_span =
        second_node - first_node;
    if (!terrain_double_is_finite(central_numerator) ||
        !terrain_double_is_finite(central_span) ||
        central_span <= 0.0) {
        return false;
    }
    const volatile double central_fraction =
        central_numerator / central_span;
    if (!terrain_double_is_finite(central_fraction) ||
        central_fraction < 0.0 || central_fraction > 1.0) {
        return false;
    }

    G1ClearanceInterval numerator = {};
    G1ClearanceInterval span = {};
    G1ClearanceInterval fraction = {};
    if (!g1_clearance_interval_subtract(
            {coordinate, coordinate},
            {first_node, first_node}, numerator) ||
        !g1_clearance_interval_subtract(
            {second_node, second_node},
            {first_node, first_node}, span) ||
        span.lower <= 0.0 ||
        !g1_clearance_interval_divide(
            numerator, span, fraction)) {
        return false;
    }

    if (fraction.upper < 0.0 || fraction.lower > 1.0) {
        return false;
    }
    fraction.lower = fraction.lower < 0.0 ? 0.0 : fraction.lower;
    fraction.upper = fraction.upper > 1.0 ? 1.0 : fraction.upper;
    if (fraction.lower > fraction.upper ||
        central_fraction < fraction.lower ||
        central_fraction > fraction.upper) {
        return false;
    }

    G1PointFraction candidate = {};
    candidate.central = central_fraction;
    candidate.enclosure = fraction;
    output = candidate;
    return true;
}

static G1PointGeometryStatus g1_clearance_point_geometry(
    const heightfield& field,
    const heightfield_cell& cell,
    vec3 point,
    G1PointGeometry& output)
{
    G1PointGeometry candidate = {};
    if (!g1_clearance_point_node(
            field.origin_x, cell.x0,
            field.cell_size, candidate.x0) ||
        !g1_clearance_point_node(
            field.origin_x, cell.x0 + 1,
            field.cell_size, candidate.x1) ||
        !g1_clearance_point_node(
            field.origin_z, cell.z0,
            field.cell_size, candidate.z0) ||
        !g1_clearance_point_node(
            field.origin_z, cell.z0 + 1,
            field.cell_size, candidate.z1) ||
        !g1_clearance_point_fraction_enclosure(
            static_cast<double>(point.x),
            candidate.x0, candidate.x1, candidate.x) ||
        !g1_clearance_point_fraction_enclosure(
            static_cast<double>(point.z),
            candidate.z0, candidate.z1, candidate.z)) {
        return G1PointGeometryArithmeticFailure;
    }

    const bool exact_materialized_tie =
        std::memcmp(
            &point.x, &point.z, sizeof(point.x)) == 0 &&
        std::memcmp(
            &candidate.x0, &candidate.z0,
            sizeof(candidate.x0)) == 0 &&
        std::memcmp(
            &candidate.x1, &candidate.z1,
            sizeof(candidate.x1)) == 0;
    if (exact_materialized_tie ||
        candidate.x.enclosure.lower >=
            candidate.z.enclosure.upper) {
        candidate.triangle_index = 0;
    } else if (candidate.x.enclosure.upper <
               candidate.z.enclosure.lower) {
        candidate.triangle_index = 1;
    } else {
        return G1PointGeometryUncertified;
    }

    output = candidate;
    return G1PointGeometryValid;
}

static bool g1_clearance_point_triangle(
    const G1PointGeometry& geometry,
    double h00,
    double h10,
    double h01,
    double h11,
    G1PointTriangle& output)
{
    if (geometry.triangle_index == 0) {
        output.x[0] = geometry.x0;
        output.z[0] = geometry.z0;
        output.height[0] = h00;
        output.x[1] = geometry.x1;
        output.z[1] = geometry.z0;
        output.height[1] = h10;
        output.x[2] = geometry.x1;
        output.z[2] = geometry.z1;
        output.height[2] = h11;
        output.index = 0;
    } else if (geometry.triangle_index == 1) {
        output.x[0] = geometry.x0;
        output.z[0] = geometry.z0;
        output.height[0] = h00;
        output.x[1] = geometry.x1;
        output.z[1] = geometry.z1;
        output.height[1] = h11;
        output.x[2] = geometry.x0;
        output.z[2] = geometry.z1;
        output.height[2] = h01;
        output.index = 1;
    } else {
        return false;
    }
    return true;
}

// Candidate kind 3 has a point-special proof source.  Its canonical promoted
// input XZ, t=0, checked closed triangle, and outward plane-height enclosure
// are authoritative.  These rounded producer-cell weights are deterministic
// public diagnostics only; capsule-patch homogeneous weights retain the
// separate exact-real normalization/feasibility contract.
static bool g1_clearance_point_weight_diagnostics(
    const G1PointGeometry& geometry,
    uint32_t triangle_index,
    double diagnostics[3])
{
    if (triangle_index == 0) {
        const volatile double first = 1.0 - geometry.x.central;
        const volatile double second =
            geometry.x.central - geometry.z.central;
        diagnostics[0] = first;
        diagnostics[1] = second;
        diagnostics[2] = geometry.z.central;
    } else if (triangle_index == 1) {
        const volatile double first = 1.0 - geometry.z.central;
        const volatile double third =
            geometry.z.central - geometry.x.central;
        diagnostics[0] = first;
        diagnostics[1] = geometry.x.central;
        diagnostics[2] = third;
    } else {
        return false;
    }
    for (int index = 0; index < 3; ++index) {
        if (!terrain_double_is_finite(diagnostics[index]) ||
            diagnostics[index] < 0.0 || diagnostics[index] > 1.0) {
            return false;
        }
        if (diagnostics[index] == 0.0) {
            diagnostics[index] = 0.0;
        }
    }
    const volatile double first_sum =
        diagnostics[0] + diagnostics[1];
    const volatile double sum = first_sum + diagnostics[2];
    return terrain_double_is_finite(sum) && sum > 0.0;
}

static bool g1_clearance_point_height(
    const G1PointGeometry& geometry,
    double h00,
    double h10,
    double h01,
    double h11,
    double& central,
    G1ClearanceInterval& enclosure)
{
    const G1ClearanceInterval tx = geometry.x.enclosure;
    const G1ClearanceInterval tz = geometry.z.enclosure;
    G1ClearanceInterval difference_x = {};
    G1ClearanceInterval x_term = {};
    G1ClearanceInterval first_sum = {};
    G1ClearanceInterval difference_z = {};
    G1ClearanceInterval z_term = {};
    const G1ClearanceInterval height00 = {h00, h00};
    if (geometry.triangle_index == 0) {
        const volatile double central_difference_x = h10 - h00;
        const volatile double central_x_term =
            geometry.x.central * central_difference_x;
        const volatile double central_first_sum =
            h00 + central_x_term;
        const volatile double central_difference_z = h11 - h10;
        const volatile double central_z_term =
            geometry.z.central * central_difference_z;
        const volatile double central_final_sum =
            central_first_sum + central_z_term;
        central = central_final_sum;
        return terrain_double_is_finite(central) &&
               g1_clearance_interval_subtract(
                   {h10, h10}, height00, difference_x) &&
               g1_clearance_interval_multiply(
                   tx, difference_x, x_term) &&
               g1_clearance_interval_add(
                   height00, x_term, first_sum) &&
               g1_clearance_interval_subtract(
                   {h11, h11}, {h10, h10}, difference_z) &&
               g1_clearance_interval_multiply(
                   tz, difference_z, z_term) &&
               g1_clearance_interval_add(
                   first_sum, z_term, enclosure) &&
               enclosure.lower <= central &&
               central <= enclosure.upper;
    }

    if (geometry.triangle_index != 1) {
        return false;
    }
    const volatile double central_difference_x = h11 - h01;
    const volatile double central_x_term =
        geometry.x.central * central_difference_x;
    const volatile double central_first_sum = h00 + central_x_term;
    const volatile double central_difference_z = h01 - h00;
    const volatile double central_z_term =
        geometry.z.central * central_difference_z;
    const volatile double central_final_sum =
        central_first_sum + central_z_term;
    central = central_final_sum;
    return terrain_double_is_finite(central) &&
           g1_clearance_interval_subtract(
               {h11, h11}, {h01, h01}, difference_x) &&
           g1_clearance_interval_multiply(
               tx, difference_x, x_term) &&
           g1_clearance_interval_add(
               height00, x_term, first_sum) &&
           g1_clearance_interval_subtract(
               {h01, h01}, height00, difference_z) &&
           g1_clearance_interval_multiply(
               tz, difference_z, z_term) &&
           g1_clearance_interval_add(
               first_sum, z_term, enclosure) &&
           enclosure.lower <= central &&
           central <= enclosure.upper;
}

static bool g1_clearance_point_float_guard(
    const G1PointTriangle& triangle,
    double& output)
{
    output = 0.0;
    double minimum_height = triangle.height[0];
    double maximum_height = triangle.height[0];
    const float positive_infinity =
        std::numeric_limits<float>::infinity();
    const float negative_infinity = -positive_infinity;
    for (int index = 0; index < 3; ++index) {
        const double height = triangle.height[index];
        const float stored_height = static_cast<float>(height);
        if (!terrain_double_is_finite(height) ||
            !terrain_float_is_normal_or_positive_zero(stored_height) ||
            static_cast<double>(stored_height) != height) {
            return false;
        }
        minimum_height = height < minimum_height
            ? height
            : minimum_height;
        maximum_height = height > maximum_height
            ? height
            : maximum_height;
        const float lower_neighbor =
            std::nextafter(stored_height, negative_infinity);
        const float upper_neighbor =
            std::nextafter(stored_height, positive_infinity);
        if (!terrain_float_is_finite(lower_neighbor) ||
            !terrain_float_is_finite(upper_neighbor)) {
            output = std::numeric_limits<double>::infinity();
            return true;
        }
        const volatile double lower_spacing =
            height - static_cast<double>(lower_neighbor);
        const volatile double upper_spacing =
            static_cast<double>(upper_neighbor) - height;
        if (!terrain_double_is_finite(lower_spacing) ||
            !terrain_double_is_finite(upper_spacing) ||
            lower_spacing <= 0.0 || upper_spacing <= 0.0) {
            return false;
        }
        output = lower_spacing > output ? lower_spacing : output;
        output = upper_spacing > output ? upper_spacing : output;
    }

    const double minimum_normal =
        static_cast<double>(std::numeric_limits<float>::min());
    if (minimum_height <= minimum_normal &&
        maximum_height >= -minimum_normal &&
        output < minimum_normal) {
        output = minimum_normal;
    }
    return terrain_double_is_finite(output) && output > 0.0;
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
    const heightfield& field,
    vec3 point,
    char* error,
    int error_capacity)
{
    const G1ClearanceProtectedRange protected_ranges[] = {
        {&output, sizeof(output)},
        {&limits, sizeof(limits)}
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
    if (!g1_clearance_budget_within(
            limits, g1_pose_clearance_budget())) {
        return g1_clearance_error(
            G1ClearanceInvalidInput,
            diagnostic.output, diagnostic.capacity,
            "G1 point clearance budget exceeds the pose factory ceiling");
    }
    if (field.version != 2 ||
        !terrain_heightfield_is_queryable(field)) {
        return g1_clearance_error(
            G1ClearanceInvalidField,
            diagnostic.output, diagnostic.capacity,
            "G1 point clearance requires a queryable G1HF/v2 field");
    }
    if (!g1_ik_vec3_is_runtime_value(point)) {
        return g1_clearance_error(
            G1ClearanceInvalidInput,
            diagnostic.output, diagnostic.capacity,
            "G1 point clearance requires runtime-normal-or-zero XYZ");
    }
    point = vec3(
        terrain_runtime_canonicalize_output(point.x),
        terrain_runtime_canonicalize_output(point.y),
        terrain_runtime_canonicalize_output(point.z));

    if (limits.maximum_point_queries < 1 ||
        limits.maximum_cells < 1 ||
        limits.maximum_primitive_triangle_pairs < 1) {
        return g1_clearance_error(
            G1ClearanceBudgetExceeded,
            diagnostic.output, diagnostic.capacity,
            "G1 point clearance work exceeds the tightened budget");
    }

    G1SurfaceSample producer = {};
    const G1SurfaceQueryStatus producer_status =
        g1_surface_query_v2(
            producer, field, point.x, point.z);
    if (producer_status == G1SurfaceQueryOutside) {
        return g1_clearance_error(
            G1ClearanceOutsideDomain,
            diagnostic.output, diagnostic.capacity,
            "G1 point clearance is outside the G1HF/v2 domain");
    }
    if (producer_status == G1SurfaceQueryInvalid) {
        heightfield_cell invalid_cell = {};
        if (!terrain_v2_locate_cell(
                field, point.x, point.z, invalid_cell)) {
            return g1_clearance_error(
                G1ClearanceArithmeticFailure,
                diagnostic.output, diagnostic.capacity,
                "G1 point producer failed after validated domain inputs");
        }
        double invalid_h00 = 0.0;
        double invalid_h10 = 0.0;
        double invalid_h01 = 0.0;
        double invalid_h11 = 0.0;
        if (!terrain_v2_cell_heights(
                field, invalid_cell,
                invalid_h00, invalid_h10,
                invalid_h01, invalid_h11)) {
            return g1_clearance_error(
                G1ClearanceInvalidField,
                diagnostic.output, diagnostic.capacity,
                "G1 point clearance encountered invalid visited heights");
        }
        return g1_clearance_error(
            G1ClearanceArithmeticFailure,
            diagnostic.output, diagnostic.capacity,
            "G1 point producer could not materialize finite surface data");
    }

    const G1ClearanceDomainStatus domain_status =
        g1_clearance_domain_contains(field, point);
    if (domain_status != G1ClearanceDomainInside) {
        return g1_clearance_error(
            domain_status == G1ClearanceDomainOutside
                ? G1ClearanceOutsideDomain
                : G1ClearanceArithmeticFailure,
            diagnostic.output, diagnostic.capacity,
            "G1 point strict domain reconstruction disagreed with producer");
    }

    heightfield_cell cell = {};
    if (!terrain_v2_locate_cell(field, point.x, point.z, cell)) {
        return g1_clearance_error(
            G1ClearanceArithmeticFailure,
            diagnostic.output, diagnostic.capacity,
            "G1 point clearance could not reconstruct its checked cell");
    }
    double h00 = 0.0;
    double h10 = 0.0;
    double h01 = 0.0;
    double h11 = 0.0;
    if (!terrain_v2_cell_heights(
            field, cell, h00, h10, h01, h11)) {
        return g1_clearance_error(
            G1ClearanceInvalidField,
            diagnostic.output, diagnostic.capacity,
            "G1 point clearance encountered invalid visited heights");
    }

    G1PointGeometry geometry = {};
    const G1PointGeometryStatus geometry_status =
        g1_clearance_point_geometry(field, cell, point, geometry);
    if (geometry_status == G1PointGeometryUncertified) {
        return g1_clearance_error(
            G1ClearanceUncertified,
            diagnostic.output, diagnostic.capacity,
            "G1 point materialized diagonal could not be certified");
    }
    if (geometry_status != G1PointGeometryValid) {
        return g1_clearance_error(
            G1ClearanceArithmeticFailure,
            diagnostic.output, diagnostic.capacity,
            "G1 point materialized coordinates could not be enclosed");
    }

    G1PointTriangle triangle = {};
    double weight_diagnostics[3] = {};
    double surface_height_diagnostic = 0.0;
    G1ClearanceInterval height_enclosure = {};
    if (!g1_clearance_point_triangle(
            geometry, h00, h10, h01, h11, triangle) ||
        !g1_clearance_point_height(
            geometry, h00, h10, h01, h11,
            surface_height_diagnostic, height_enclosure)) {
        return g1_clearance_error(
            G1ClearanceArithmeticFailure,
            diagnostic.output, diagnostic.capacity,
            "G1 point clearance could not enclose its affine triangle");
    }

    double float_guard = 0.0;
    if (!g1_clearance_point_float_guard(triangle, float_guard)) {
        return g1_clearance_error(
            G1ClearanceArithmeticFailure,
            diagnostic.output, diagnostic.capacity,
            "G1 point clearance could not derive its output guard");
    }
    if (!terrain_double_is_finite(float_guard) ||
        float_guard > G1ClearanceMaximumCertificateWidthM) {
        return g1_clearance_error(
            G1ClearanceUncertified,
            diagnostic.output, diagnostic.capacity,
            "G1 point mandatory float-output guard is too wide");
    }

    const G1ClearanceInterval body_y = {
        static_cast<double>(point.y),
        static_cast<double>(point.y)
    };
    G1ClearanceInterval clearance = {};
    if (!g1_clearance_interval_subtract(
            body_y, height_enclosure, clearance)) {
        return g1_clearance_error(
            G1ClearanceArithmeticFailure,
            diagnostic.output, diagnostic.capacity,
            "G1 point clearance subtraction could not be enclosed");
    }
    const volatile double guarded_value =
        clearance.lower - float_guard;
    double guarded_lower = 0.0;
    if (!g1_clearance_outward_lower(
            guarded_value, guarded_lower)) {
        return g1_clearance_error(
            G1ClearanceArithmeticFailure,
            diagnostic.output, diagnostic.capacity,
            "G1 point guarded lower bound is nonfinite");
    }

    const volatile double producer_clearance =
        static_cast<double>(point.y) -
        static_cast<double>(producer.height);
    double producer_lower = 0.0;
    if (!g1_clearance_outward_lower(
            producer_clearance, producer_lower)) {
        return g1_clearance_error(
            G1ClearanceArithmeticFailure,
            diagnostic.output, diagnostic.capacity,
            "G1 point producer lower cap is nonfinite");
    }
    if (producer_lower < guarded_lower) {
        guarded_lower = producer_lower;
    }
    if (!terrain_double_is_finite(guarded_lower) ||
        !terrain_double_is_finite(clearance.upper) ||
        guarded_lower > clearance.upper) {
        return g1_clearance_error(
            G1ClearanceArithmeticFailure,
            diagnostic.output, diagnostic.capacity,
            "G1 point certificate endpoints are invalid");
    }

    const volatile double rounded_width =
        clearance.upper - guarded_lower;
    double width_upper = 0.0;
    if (!g1_clearance_outward_upper(
            rounded_width, width_upper)) {
        return g1_clearance_error(
            G1ClearanceArithmeticFailure,
            diagnostic.output, diagnostic.capacity,
            "G1 point certificate width is nonfinite");
    }
    if (width_upper > G1ClearanceMaximumCertificateWidthM) {
        return g1_clearance_error(
            G1ClearanceUncertified,
            diagnostic.output, diagnostic.capacity,
            "G1 point certificate exceeds the required width");
    }

    if (!g1_clearance_point_weight_diagnostics(
            geometry, triangle.index, weight_diagnostics)) {
        return g1_clearance_error(
            G1ClearanceArithmeticFailure,
            diagnostic.output, diagnostic.capacity,
            "G1 point weight diagnostics could not be materialized");
    }

    // The point-special proof is the canonical point XZ and the strict
    // selected-triangle height enclosure above.  The XYZ and weight members
    // below expose deterministic diagnostics; they are not read back as the
    // feasibility source for candidate_kind=3.
    G1ClearanceResult candidate = {};
    candidate.lower_bound_m = guarded_lower;
    candidate.witness_upper_m = clearance.upper;
    candidate.witness.body_x = static_cast<double>(point.x);
    candidate.witness.body_y = static_cast<double>(point.y);
    candidate.witness.body_z = static_cast<double>(point.z);
    candidate.witness.surface_x = static_cast<double>(point.x);
    candidate.witness.surface_y = surface_height_diagnostic;
    candidate.witness.surface_z = static_cast<double>(point.z);
    candidate.witness.segment_parameter = 0.0;
    candidate.witness.terrain_weight_0 = weight_diagnostics[0];
    candidate.witness.terrain_weight_1 = weight_diagnostics[1];
    candidate.witness.terrain_weight_2 = weight_diagnostics[2];
    candidate.witness.primitive_index = 0;
    candidate.witness.cell_x = cell.x0;
    candidate.witness.cell_z = cell.z0;
    candidate.witness.terrain_triangle_index = triangle.index;
    candidate.witness.patch_index = 0;
    candidate.witness.candidate_kind = 3;
    candidate.witness.candidate_subindex = 0;
    candidate.work.point_queries = 1;
    candidate.work.cells_visited = 1;
    candidate.work.primitive_triangle_pairs = 1;
    output = candidate;
    return G1ClearanceOk;
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
