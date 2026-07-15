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

static bool g1_clearance_interval_negate(
    const G1ClearanceInterval& input,
    G1ClearanceInterval& output)
{
    if (!terrain_double_is_finite(input.lower) ||
        !terrain_double_is_finite(input.upper) ||
        input.lower > input.upper) {
        return false;
    }
    const volatile double lower = -input.upper;
    const volatile double upper = -input.lower;
    if (!terrain_double_is_finite(lower) ||
        !terrain_double_is_finite(upper)) {
        return false;
    }
    output.lower = lower == 0.0 ? 0.0 : lower;
    output.upper = upper == 0.0 ? 0.0 : upper;
    return output.lower <= output.upper;
}

static bool g1_clearance_interval_square(
    const G1ClearanceInterval& input,
    G1ClearanceInterval& output)
{
    if (!terrain_double_is_finite(input.lower) ||
        !terrain_double_is_finite(input.upper) ||
        input.lower > input.upper) {
        return false;
    }
    if (input.lower <= 0.0 && input.upper >= 0.0) {
        const volatile double lower_square = input.lower * input.lower;
        const volatile double upper_square = input.upper * input.upper;
        const double maximum = lower_square > upper_square
            ? lower_square
            : upper_square;
        output.lower = 0.0;
        return g1_clearance_outward_upper(maximum, output.upper);
    }
    return g1_clearance_interval_multiply(input, input, output);
}

static bool g1_clearance_interval_sqrt(
    const G1ClearanceInterval& input,
    G1ClearanceInterval& output)
{
    if (!terrain_double_is_finite(input.lower) ||
        !terrain_double_is_finite(input.upper) ||
        input.lower > input.upper || input.upper < 0.0) {
        return false;
    }
    const double clamped_lower = input.lower <= 0.0
        ? 0.0
        : input.lower;
    const volatile double lower = std::sqrt(clamped_lower);
    const volatile double upper = std::sqrt(input.upper);
    if (!terrain_double_is_finite(lower) ||
        !terrain_double_is_finite(upper)) {
        return false;
    }
    output.lower = lower == 0.0
        ? 0.0
        : std::nextafter(
            static_cast<double>(lower),
            -std::numeric_limits<double>::infinity());
    return terrain_double_is_finite(output.lower) &&
           g1_clearance_outward_upper(upper, output.upper) &&
           output.lower <= output.upper;
}

static bool g1_clearance_interval_sum3(
    const G1ClearanceInterval& first,
    const G1ClearanceInterval& second,
    const G1ClearanceInterval& third,
    G1ClearanceInterval& output)
{
    G1ClearanceInterval partial = {};
    return g1_clearance_interval_add(first, second, partial) &&
           g1_clearance_interval_add(partial, third, output);
}

struct G1ClearanceExpansion2
{
    double high;
    double low;
};

static bool g1_clearance_two_sum(
    double left,
    double right,
    G1ClearanceExpansion2& output)
{
    const volatile double sum = left + right;
    if (!terrain_double_is_finite(sum)) {
        return false;
    }
    const volatile double right_virtual = sum - left;
    const volatile double left_virtual = sum - right_virtual;
    const volatile double right_roundoff = right - right_virtual;
    const volatile double left_roundoff = left - left_virtual;
    const volatile double error = left_roundoff + right_roundoff;
    if (!terrain_double_is_finite(error)) {
        return false;
    }
    output.high = sum == 0.0 ? 0.0 : sum;
    output.low = error == 0.0 ? 0.0 : error;
    return true;
}

static bool g1_clearance_two_diff(
    double left,
    double right,
    G1ClearanceExpansion2& output)
{
    const volatile double difference = left - right;
    if (!terrain_double_is_finite(difference)) {
        return false;
    }
    const volatile double right_virtual = left - difference;
    const volatile double left_virtual = difference + right_virtual;
    const volatile double right_roundoff = right_virtual - right;
    const volatile double left_roundoff = left - left_virtual;
    const volatile double error = left_roundoff + right_roundoff;
    if (!terrain_double_is_finite(error)) {
        return false;
    }
    output.high = difference == 0.0 ? 0.0 : difference;
    output.low = error == 0.0 ? 0.0 : error;
    return true;
}

static bool g1_clearance_expansion_interval(
    const G1ClearanceExpansion2& expansion,
    G1ClearanceInterval& output)
{
    const volatile double central = expansion.high + expansion.low;
    return g1_clearance_outward_lower(central, output.lower) &&
           g1_clearance_outward_upper(central, output.upper) &&
           output.lower <= output.upper;
}

static int g1_clearance_expansion_compare_double(
    const G1ClearanceExpansion2& expansion,
    double value)
{
    G1ClearanceExpansion2 difference = {};
    if (!terrain_double_is_finite(expansion.high) ||
        !terrain_double_is_finite(expansion.low) ||
        !terrain_double_is_finite(value) ||
        !g1_clearance_two_diff(expansion.high, value, difference)) {
        return 2;
    }

    // Build an exact nonoverlapping expansion of
    // (high-value)+difference.low+expansion.low.  The sign of the largest
    // nonzero component is the exact-real comparison result.
    double terms[4] = {difference.low, difference.high, 0.0, 0.0};
    int count = difference.low == 0.0 ? 0 : 1;
    if (difference.high != 0.0) {
        terms[count++] = difference.high;
    }
    double carry = expansion.low;
    double grown[4] = {};
    int grown_count = 0;
    for (int index = 0; index < count; ++index) {
        G1ClearanceExpansion2 sum = {};
        if (!g1_clearance_two_sum(carry, terms[index], sum)) {
            return 2;
        }
        if (sum.low != 0.0) {
            grown[grown_count++] = sum.low;
        }
        carry = sum.high;
    }
    if (carry != 0.0 || grown_count == 0) {
        grown[grown_count++] = carry;
    }
    for (int index = grown_count - 1; index >= 0; --index) {
        if (grown[index] < 0.0) return -1;
        if (grown[index] > 0.0) return 1;
    }
    return 0;
}

struct G1EndpointSourceKey
{
    uint32_t source_kind;
    uint32_t primitive_index;
    uint32_t semantic_ordinal;
    uint32_t original_x_bits;
    uint32_t original_y_bits;
    uint32_t original_z_bits;
};

struct G1ExactY
{
    double terms[4];
    uint32_t count;
    G1ClearanceInterval enclosure;
};

struct G1CertifiedEndpoint
{
    double x;
    double z;
    G1ExactY y;
    G1EndpointSourceKey source_key;
};

static uint32_t g1_clearance_float_total_order(uint32_t bits)
{
    return (bits & UINT32_C(0x80000000)) != 0
        ? ~bits
        : bits ^ UINT32_C(0x80000000);
}

static bool g1_clearance_endpoint_key_less(
    const G1EndpointSourceKey& left,
    const G1EndpointSourceKey& right)
{
    if (left.source_kind != right.source_kind) {
        return left.source_kind < right.source_kind;
    }
    if (left.primitive_index != right.primitive_index) {
        return left.primitive_index < right.primitive_index;
    }
    if (left.semantic_ordinal != right.semantic_ordinal) {
        return left.semantic_ordinal < right.semantic_ordinal;
    }
    if (left.original_x_bits != right.original_x_bits) {
        return left.original_x_bits < right.original_x_bits;
    }
    if (left.original_y_bits != right.original_y_bits) {
        return left.original_y_bits < right.original_y_bits;
    }
    return left.original_z_bits < right.original_z_bits;
}

static bool g1_clearance_public_endpoint_less(vec3 left, vec3 right)
{
    const uint32_t left_order[3] = {
        g1_clearance_float_total_order(terrain_float_bits(left.x)),
        g1_clearance_float_total_order(terrain_float_bits(left.y)),
        g1_clearance_float_total_order(terrain_float_bits(left.z))
    };
    const uint32_t right_order[3] = {
        g1_clearance_float_total_order(terrain_float_bits(right.x)),
        g1_clearance_float_total_order(terrain_float_bits(right.y)),
        g1_clearance_float_total_order(terrain_float_bits(right.z))
    };
    for (int index = 0; index < 3; ++index) {
        if (left_order[index] != right_order[index]) {
            return left_order[index] < right_order[index];
        }
    }
    return false;
}

static bool g1_clearance_make_public_endpoint(
    vec3 value,
    uint32_t semantic_ordinal,
    G1CertifiedEndpoint& output)
{
    if (!g1_ik_vec3_is_runtime_value(value)) {
        return false;
    }
    value = vec3(
        terrain_runtime_canonicalize_output(value.x),
        terrain_runtime_canonicalize_output(value.y),
        terrain_runtime_canonicalize_output(value.z));
    G1CertifiedEndpoint candidate = {};
    candidate.x = static_cast<double>(value.x);
    candidate.z = static_cast<double>(value.z);
    candidate.y.terms[0] = static_cast<double>(value.y);
    candidate.y.count = 1;
    candidate.y.enclosure = {
        static_cast<double>(value.y),
        static_cast<double>(value.y)
    };
    candidate.source_key.source_kind = 0;
    candidate.source_key.primitive_index = 0;
    candidate.source_key.semantic_ordinal = semantic_ordinal;
    candidate.source_key.original_x_bits = terrain_float_bits(value.x);
    candidate.source_key.original_y_bits = terrain_float_bits(value.y);
    candidate.source_key.original_z_bits = terrain_float_bits(value.z);
    output = candidate;
    return true;
}

static bool g1_clearance_make_public_endpoints(
    vec3 endpoint_a,
    vec3 endpoint_b,
    G1CertifiedEndpoint output[2])
{
    endpoint_a = vec3(
        terrain_runtime_canonicalize_output(endpoint_a.x),
        terrain_runtime_canonicalize_output(endpoint_a.y),
        terrain_runtime_canonicalize_output(endpoint_a.z));
    endpoint_b = vec3(
        terrain_runtime_canonicalize_output(endpoint_b.x),
        terrain_runtime_canonicalize_output(endpoint_b.y),
        terrain_runtime_canonicalize_output(endpoint_b.z));
    if (g1_clearance_public_endpoint_less(endpoint_b, endpoint_a)) {
        const vec3 temporary = endpoint_a;
        endpoint_a = endpoint_b;
        endpoint_b = temporary;
    }
    return g1_clearance_make_public_endpoint(endpoint_a, 0, output[0]) &&
           g1_clearance_make_public_endpoint(endpoint_b, 1, output[1]);
}

struct G1ClearanceCellSpan
{
    int minimum_x;
    int maximum_x;
    int minimum_z;
    int maximum_z;
    uint64_t cell_count;
};

static bool g1_clearance_checked_multiply_u64(
    uint64_t left,
    uint64_t right,
    uint64_t& output)
{
    if (left != 0 && right > UINT64_MAX / left) {
        return false;
    }
    output = left * right;
    return true;
}

static G1ClearanceDomainStatus g1_clearance_footprint_axis(
    double endpoint_a,
    double endpoint_b,
    double radius,
    double origin,
    int node_count,
    double cell_size,
    int& minimum_cell,
    int& maximum_cell)
{
    if (!terrain_double_is_finite(endpoint_a) ||
        !terrain_double_is_finite(endpoint_b) ||
        !terrain_double_is_finite(radius) || radius <= 0.0 ||
        !terrain_double_is_finite(origin) ||
        !terrain_double_is_finite(cell_size) || cell_size <= 0.0 ||
        node_count < 2) {
        return G1ClearanceDomainArithmeticFailure;
    }
    const double minimum_endpoint = endpoint_a < endpoint_b
        ? endpoint_a
        : endpoint_b;
    const double maximum_endpoint = endpoint_a > endpoint_b
        ? endpoint_a
        : endpoint_b;
    G1ClearanceExpansion2 minimum = {};
    G1ClearanceExpansion2 maximum = {};
    if (!g1_clearance_two_diff(minimum_endpoint, radius, minimum) ||
        !g1_clearance_two_sum(maximum_endpoint, radius, maximum)) {
        return G1ClearanceDomainArithmeticFailure;
    }
    const volatile double maximum_product =
        static_cast<double>(node_count - 1) * cell_size;
    const volatile double domain_maximum = origin + maximum_product;
    if (!terrain_double_is_finite(maximum_product) ||
        !terrain_double_is_finite(domain_maximum)) {
        return G1ClearanceDomainArithmeticFailure;
    }
    const int minimum_comparison =
        g1_clearance_expansion_compare_double(minimum, origin);
    const int maximum_comparison =
        g1_clearance_expansion_compare_double(maximum, domain_maximum);
    if (minimum_comparison == 2 || maximum_comparison == 2) {
        return G1ClearanceDomainArithmeticFailure;
    }
    if (minimum_comparison < 0 || maximum_comparison > 0) {
        return G1ClearanceDomainOutside;
    }

    G1ClearanceInterval minimum_interval = {};
    G1ClearanceInterval maximum_interval = {};
    G1ClearanceInterval minimum_relative = {};
    G1ClearanceInterval maximum_relative = {};
    G1ClearanceInterval minimum_quotient = {};
    G1ClearanceInterval maximum_quotient = {};
    if (!g1_clearance_expansion_interval(minimum, minimum_interval) ||
        !g1_clearance_expansion_interval(maximum, maximum_interval) ||
        !g1_clearance_interval_subtract(
            minimum_interval, {origin, origin}, minimum_relative) ||
        !g1_clearance_interval_subtract(
            maximum_interval, {origin, origin}, maximum_relative) ||
        !g1_clearance_interval_divide(
            minimum_relative, {cell_size, cell_size},
            minimum_quotient) ||
        !g1_clearance_interval_divide(
            maximum_relative, {cell_size, cell_size},
            maximum_quotient)) {
        return G1ClearanceDomainArithmeticFailure;
    }
    const volatile double first_floor =
        std::floor(minimum_quotient.lower);
    const volatile double last_floor =
        std::floor(maximum_quotient.upper);
    if (!terrain_double_is_finite(first_floor) ||
        !terrain_double_is_finite(last_floor) ||
        first_floor < -1.0 ||
        last_floor > static_cast<double>(node_count)) {
        return G1ClearanceDomainArithmeticFailure;
    }
    const double clamped_first = first_floor < 0.0
        ? 0.0
        : (first_floor > static_cast<double>(node_count - 2)
            ? static_cast<double>(node_count - 2)
            : first_floor);
    const double clamped_last = last_floor < 0.0
        ? 0.0
        : (last_floor > static_cast<double>(node_count - 2)
            ? static_cast<double>(node_count - 2)
            : last_floor);
    minimum_cell = static_cast<int>(clamped_first);
    maximum_cell = static_cast<int>(clamped_last);
    return minimum_cell <= maximum_cell
        ? G1ClearanceDomainInside
        : G1ClearanceDomainArithmeticFailure;
}

static G1ClearanceDomainStatus g1_clearance_footprint_span(
    const heightfield& field,
    const G1CertifiedEndpoint endpoints[2],
    double radius,
    G1ClearanceCellSpan& output)
{
    G1ClearanceCellSpan candidate = {};
    const G1ClearanceDomainStatus x_status =
        g1_clearance_footprint_axis(
            endpoints[0].x, endpoints[1].x, radius,
            static_cast<double>(field.origin_x), field.nx,
            static_cast<double>(field.cell_size),
            candidate.minimum_x, candidate.maximum_x);
    if (x_status != G1ClearanceDomainInside) {
        return x_status;
    }
    const G1ClearanceDomainStatus z_status =
        g1_clearance_footprint_axis(
            endpoints[0].z, endpoints[1].z, radius,
            static_cast<double>(field.origin_z), field.nz,
            static_cast<double>(field.cell_size),
            candidate.minimum_z, candidate.maximum_z);
    if (z_status != G1ClearanceDomainInside) {
        return z_status;
    }
    const uint64_t x_count = static_cast<uint64_t>(
        candidate.maximum_x - candidate.minimum_x + 1);
    const uint64_t z_count = static_cast<uint64_t>(
        candidate.maximum_z - candidate.minimum_z + 1);
    if (!g1_clearance_checked_multiply_u64(
            x_count, z_count, candidate.cell_count)) {
        return G1ClearanceDomainArithmeticFailure;
    }
    output = candidate;
    return G1ClearanceDomainInside;
}

struct G1CapsuleTriangle
{
    double x[3];
    double z[3];
    double height[3];
    int cell_x;
    int cell_z;
    uint32_t triangle_index;
};

struct G1CapsulePatchVertex
{
    double x;
    double z;
    double y;
    G1ClearanceExpansion2 x_exact;
    G1ClearanceExpansion2 z_exact;
    G1ClearanceInterval x_enclosure;
    G1ClearanceInterval z_enclosure;
    G1ClearanceInterval y_enclosure;
    uint32_t endpoint_index;
    uint32_t terrain_vertex_index;
};

struct G1CapsulePatch
{
    G1CapsulePatchVertex vertex[3];
    uint32_t patch_index;
};

struct G1CapsuleWitnessCandidate
{
    double upper;
    G1ClearanceWitness witness;
};

struct G1CapsulePatchResult
{
    bool has_lower;
    double lower;
    bool has_witness;
    G1CapsuleWitnessCandidate witness;
    bool used_relaxed_lower;
};

static bool g1_clearance_interval_minimum(
    const G1ClearanceInterval& left,
    const G1ClearanceInterval& right,
    G1ClearanceInterval& output)
{
    if (!terrain_double_is_finite(left.lower) ||
        !terrain_double_is_finite(left.upper) ||
        !terrain_double_is_finite(right.lower) ||
        !terrain_double_is_finite(right.upper) ||
        left.lower > left.upper || right.lower > right.upper) {
        return false;
    }
    output.lower = left.lower < right.lower
        ? left.lower
        : right.lower;
    output.upper = left.upper < right.upper
        ? left.upper
        : right.upper;
    return output.lower <= output.upper;
}

static bool g1_clearance_interval_maximum(
    const G1ClearanceInterval& left,
    const G1ClearanceInterval& right,
    G1ClearanceInterval& output)
{
    if (!terrain_double_is_finite(left.lower) ||
        !terrain_double_is_finite(left.upper) ||
        !terrain_double_is_finite(right.lower) ||
        !terrain_double_is_finite(right.upper) ||
        left.lower > left.upper || right.lower > right.upper) {
        return false;
    }
    output.lower = left.lower > right.lower
        ? left.lower
        : right.lower;
    output.upper = left.upper > right.upper
        ? left.upper
        : right.upper;
    return output.lower <= output.upper;
}

static bool g1_clearance_exact_y_sum(
    const G1ExactY& value,
    double& output)
{
    if (value.count == 0 || value.count > 4) {
        return false;
    }
    volatile double sum = 0.0;
    for (uint32_t index = 0; index < value.count; ++index) {
        sum = sum + value.terms[index];
        if (!terrain_double_is_finite(sum)) {
            return false;
        }
    }
    output = sum == 0.0 ? 0.0 : sum;
    return true;
}

static bool g1_clearance_make_capsule_triangle(
    const heightfield& field,
    int cell_x,
    int cell_z,
    uint32_t triangle_index,
    double h00,
    double h10,
    double h01,
    double h11,
    G1CapsuleTriangle& output)
{
    double x0 = 0.0;
    double x1 = 0.0;
    double z0 = 0.0;
    double z1 = 0.0;
    if (!g1_clearance_point_node(
            field.origin_x, cell_x, field.cell_size, x0) ||
        !g1_clearance_point_node(
            field.origin_x, cell_x + 1, field.cell_size, x1) ||
        !g1_clearance_point_node(
            field.origin_z, cell_z, field.cell_size, z0) ||
        !g1_clearance_point_node(
            field.origin_z, cell_z + 1, field.cell_size, z1)) {
        return false;
    }
    G1CapsuleTriangle candidate = {};
    candidate.cell_x = cell_x;
    candidate.cell_z = cell_z;
    candidate.triangle_index = triangle_index;
    if (triangle_index == 0) {
        candidate.x[0] = x0;
        candidate.z[0] = z0;
        candidate.height[0] = h00;
        candidate.x[1] = x1;
        candidate.z[1] = z0;
        candidate.height[1] = h10;
        candidate.x[2] = x1;
        candidate.z[2] = z1;
        candidate.height[2] = h11;
    } else if (triangle_index == 1) {
        candidate.x[0] = x0;
        candidate.z[0] = z0;
        candidate.height[0] = h00;
        candidate.x[1] = x1;
        candidate.z[1] = z1;
        candidate.height[1] = h11;
        candidate.x[2] = x0;
        candidate.z[2] = z1;
        candidate.height[2] = h01;
    } else {
        return false;
    }
    output = candidate;
    return true;
}

static bool g1_clearance_make_patch_vertex(
    const G1CertifiedEndpoint& endpoint,
    uint32_t endpoint_index,
    const G1CapsuleTriangle& triangle,
    uint32_t terrain_vertex_index,
    G1CapsulePatchVertex& output)
{
    if (endpoint_index > 1 || terrain_vertex_index > 2) {
        return false;
    }
    double endpoint_y = 0.0;
    if (!g1_clearance_exact_y_sum(endpoint.y, endpoint_y)) {
        return false;
    }
    const double terrain_x = triangle.x[terrain_vertex_index];
    const double terrain_z = triangle.z[terrain_vertex_index];
    const double terrain_y = triangle.height[terrain_vertex_index];
    const volatile double x = endpoint.x - terrain_x;
    const volatile double z = endpoint.z - terrain_z;
    const volatile double y = endpoint_y - terrain_y;
    G1CapsulePatchVertex candidate = {};
    if (!terrain_double_is_finite(x) ||
        !terrain_double_is_finite(z) ||
        !terrain_double_is_finite(y) ||
        !g1_clearance_two_diff(
            endpoint.x, terrain_x, candidate.x_exact) ||
        !g1_clearance_two_diff(
            endpoint.z, terrain_z, candidate.z_exact) ||
        !g1_clearance_interval_subtract(
            {endpoint.x, endpoint.x},
            {terrain_x, terrain_x}, candidate.x_enclosure) ||
        !g1_clearance_interval_subtract(
            {endpoint.z, endpoint.z},
            {terrain_z, terrain_z}, candidate.z_enclosure) ||
        !g1_clearance_interval_subtract(
            endpoint.y.enclosure,
            {terrain_y, terrain_y}, candidate.y_enclosure)) {
        return false;
    }
    candidate.x = x == 0.0 ? 0.0 : x;
    candidate.z = z == 0.0 ? 0.0 : z;
    candidate.y = y == 0.0 ? 0.0 : y;
    candidate.endpoint_index = endpoint_index;
    candidate.terrain_vertex_index = terrain_vertex_index;
    output = candidate;
    return true;
}

static bool g1_clearance_expansion_same(
    const G1ClearanceExpansion2& left,
    const G1ClearanceExpansion2& right)
{
    return left.high == right.high && left.low == right.low;
}

static bool g1_clearance_patch_vertices_same_xz(
    const G1CapsulePatchVertex& left,
    const G1CapsulePatchVertex& right)
{
    return g1_clearance_expansion_same(left.x_exact, right.x_exact) &&
           g1_clearance_expansion_same(left.z_exact, right.z_exact);
}

static bool g1_clearance_make_capsule_patch(
    const G1CertifiedEndpoint endpoints[2],
    const G1CapsuleTriangle& triangle,
    uint32_t patch_index,
    G1CapsulePatch& output)
{
    static const uint8_t definitions[8][3][2] = {
        {{0, 0}, {0, 1}, {0, 2}},
        {{1, 0}, {1, 1}, {1, 2}},
        {{0, 0}, {0, 1}, {1, 1}},
        {{0, 0}, {1, 1}, {1, 0}},
        {{0, 1}, {0, 2}, {1, 2}},
        {{0, 1}, {1, 2}, {1, 1}},
        {{0, 2}, {0, 0}, {1, 0}},
        {{0, 2}, {1, 0}, {1, 2}}
    };
    if (patch_index >= 8) {
        return false;
    }
    G1CapsulePatch candidate = {};
    candidate.patch_index = patch_index;
    for (uint32_t index = 0; index < 3; ++index) {
        const uint32_t endpoint_index =
            definitions[patch_index][index][0];
        const uint32_t terrain_index =
            definitions[patch_index][index][1];
        if (!g1_clearance_make_patch_vertex(
                endpoints[endpoint_index], endpoint_index,
                triangle, terrain_index,
                candidate.vertex[index])) {
            return false;
        }
    }
    output = candidate;
    return true;
}

static bool g1_clearance_witness_key_less(
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

static bool g1_clearance_patch_combination(
    const G1CapsulePatch& patch,
    const double homogeneous[3],
    const G1ClearanceInterval& homogeneous_sum,
    G1ClearanceInterval& x,
    G1ClearanceInterval& y,
    G1ClearanceInterval& z)
{
    G1ClearanceInterval x_terms[3] = {};
    G1ClearanceInterval y_terms[3] = {};
    G1ClearanceInterval z_terms[3] = {};
    for (int index = 0; index < 3; ++index) {
        if (!terrain_double_is_finite(homogeneous[index]) ||
            homogeneous[index] < 0.0 ||
            !g1_clearance_interval_multiply(
                {homogeneous[index], homogeneous[index]},
                patch.vertex[index].x_enclosure, x_terms[index]) ||
            !g1_clearance_interval_multiply(
                {homogeneous[index], homogeneous[index]},
                patch.vertex[index].y_enclosure, y_terms[index]) ||
            !g1_clearance_interval_multiply(
                {homogeneous[index], homogeneous[index]},
                patch.vertex[index].z_enclosure, z_terms[index])) {
            return false;
        }
    }
    G1ClearanceInterval x_numerator = {};
    G1ClearanceInterval y_numerator = {};
    G1ClearanceInterval z_numerator = {};
    return homogeneous_sum.lower > 0.0 &&
           g1_clearance_interval_sum3(
               x_terms[0], x_terms[1], x_terms[2], x_numerator) &&
           g1_clearance_interval_sum3(
               y_terms[0], y_terms[1], y_terms[2], y_numerator) &&
           g1_clearance_interval_sum3(
               z_terms[0], z_terms[1], z_terms[2], z_numerator) &&
           g1_clearance_interval_divide(
               x_numerator, homogeneous_sum, x) &&
           g1_clearance_interval_divide(
               y_numerator, homogeneous_sum, y) &&
           g1_clearance_interval_divide(
               z_numerator, homogeneous_sum, z);
}

static bool g1_clearance_patch_witness(
    const G1CertifiedEndpoint endpoints[2],
    const G1CapsuleTriangle& triangle,
    const G1CapsulePatch& patch,
    const double homogeneous[3],
    double radius,
    uint32_t primitive_index,
    uint32_t candidate_kind,
    uint32_t candidate_subindex,
    G1CapsuleWitnessCandidate& output)
{
    volatile double sum = 0.0;
    for (int index = 0; index < 3; ++index) {
        if (!terrain_double_is_finite(homogeneous[index]) ||
            homogeneous[index] < 0.0) {
            return false;
        }
        sum = sum + homogeneous[index];
    }
    if (!terrain_double_is_finite(sum) || sum <= 0.0) {
        return false;
    }
    G1ClearanceInterval first_sum = {};
    G1ClearanceInterval homogeneous_sum = {};
    if (!g1_clearance_interval_add(
            {homogeneous[0], homogeneous[0]},
            {homogeneous[1], homogeneous[1]}, first_sum) ||
        !g1_clearance_interval_add(
            first_sum,
            {homogeneous[2], homogeneous[2]}, homogeneous_sum) ||
        homogeneous_sum.lower <= 0.0) {
        return false;
    }
    double normalized[3] = {};
    for (int index = 0; index < 3; ++index) {
        const volatile double weight = homogeneous[index] / sum;
        if (!terrain_double_is_finite(weight) ||
            weight < 0.0 || weight > 1.0) {
            return false;
        }
        normalized[index] = weight == 0.0 ? 0.0 : weight;
    }

    G1ClearanceInterval difference_x = {};
    G1ClearanceInterval difference_y = {};
    G1ClearanceInterval difference_z = {};
    G1ClearanceInterval x_square = {};
    G1ClearanceInterval z_square = {};
    G1ClearanceInterval rho_square = {};
    G1ClearanceInterval radius_square = {};
    if (!g1_clearance_patch_combination(
            patch, homogeneous, homogeneous_sum,
            difference_x, difference_y, difference_z) ||
        !g1_clearance_interval_square(difference_x, x_square) ||
        !g1_clearance_interval_square(difference_z, z_square) ||
        !g1_clearance_interval_add(
            x_square, z_square, rho_square) ||
        !g1_clearance_interval_square(
            {radius, radius}, radius_square) ||
        rho_square.upper > radius_square.lower) {
        return false;
    }
    const volatile double allowance_value =
        radius_square.lower - rho_square.upper;
    double allowance_lower = 0.0;
    if (!g1_clearance_outward_lower(
            allowance_value, allowance_lower) ||
        allowance_lower < 0.0) {
        allowance_lower = 0.0;
    }
    volatile double magnitude = std::sqrt(allowance_lower);
    if (!terrain_double_is_finite(magnitude)) {
        return false;
    }
    if (magnitude > 0.0) {
        magnitude = std::nextafter(
            static_cast<double>(magnitude), 0.0);
    }
    bool disk_verified = false;
    for (int attempt = 0; attempt < 64; ++attempt) {
        G1ClearanceInterval magnitude_square = {};
        G1ClearanceInterval disk_sum = {};
        if (!g1_clearance_interval_square(
                {magnitude, magnitude}, magnitude_square) ||
            !g1_clearance_interval_add(
                rho_square, magnitude_square, disk_sum)) {
            return false;
        }
        if (disk_sum.upper <= radius_square.lower) {
            disk_verified = true;
            break;
        }
        if (magnitude == 0.0) {
            break;
        }
        magnitude = std::nextafter(
            static_cast<double>(magnitude), 0.0);
    }
    if (!disk_verified) {
        return false;
    }
    const volatile double objective_value =
        difference_y.upper - magnitude;
    double objective_upper = 0.0;
    if (!g1_clearance_outward_upper(
            objective_value, objective_upper)) {
        return false;
    }

    // The rounded normalized weights and endpoint sums below are public
    // diagnostics only.  Feasibility and witness_upper above come solely
    // from the homogeneous h_i/S interval reconstruction, including each
    // endpoint Y expansion already carried by the patch vertices.
    double endpoint_y[2] = {};
    if (!g1_clearance_exact_y_sum(endpoints[0].y, endpoint_y[0]) ||
        !g1_clearance_exact_y_sum(endpoints[1].y, endpoint_y[1])) {
        return false;
    }
    double terrain_weights[3] = {};
    volatile double segment_parameter = 0.0;
    for (int index = 0; index < 3; ++index) {
        terrain_weights[patch.vertex[index].terrain_vertex_index] +=
            normalized[index];
        segment_parameter = segment_parameter +
            normalized[index] *
            static_cast<double>(patch.vertex[index].endpoint_index);
    }
    if (!terrain_double_is_finite(segment_parameter) ||
        segment_parameter < 0.0 || segment_parameter > 1.0) {
        return false;
    }
    const volatile double endpoint_one_weight = segment_parameter;
    const volatile double endpoint_zero_weight =
        1.0 - endpoint_one_weight;
    const volatile double center_y =
        endpoint_zero_weight * endpoint_y[0] +
        endpoint_one_weight * endpoint_y[1];
    volatile double surface_x = 0.0;
    volatile double surface_y = 0.0;
    volatile double surface_z = 0.0;
    for (int index = 0; index < 3; ++index) {
        surface_x = surface_x +
            terrain_weights[index] * triangle.x[index];
        surface_y = surface_y +
            terrain_weights[index] * triangle.height[index];
        surface_z = surface_z +
            terrain_weights[index] * triangle.z[index];
    }
    const volatile double body_y = center_y - magnitude;
    if (!terrain_double_is_finite(center_y) ||
        !terrain_double_is_finite(surface_x) ||
        !terrain_double_is_finite(surface_y) ||
        !terrain_double_is_finite(surface_z) ||
        !terrain_double_is_finite(body_y)) {
        return false;
    }

    G1CapsuleWitnessCandidate candidate = {};
    candidate.upper = objective_upper;
    candidate.witness.body_x = surface_x == 0.0 ? 0.0 : surface_x;
    candidate.witness.body_y = body_y == 0.0 ? 0.0 : body_y;
    candidate.witness.body_z = surface_z == 0.0 ? 0.0 : surface_z;
    candidate.witness.surface_x = candidate.witness.body_x;
    candidate.witness.surface_y =
        surface_y == 0.0 ? 0.0 : surface_y;
    candidate.witness.surface_z = candidate.witness.body_z;
    candidate.witness.segment_parameter =
        segment_parameter == 0.0 ? 0.0 : segment_parameter;
    candidate.witness.terrain_weight_0 = terrain_weights[0];
    candidate.witness.terrain_weight_1 = terrain_weights[1];
    candidate.witness.terrain_weight_2 = terrain_weights[2];
    candidate.witness.primitive_index = primitive_index;
    candidate.witness.cell_x = triangle.cell_x;
    candidate.witness.cell_z = triangle.cell_z;
    candidate.witness.terrain_triangle_index =
        triangle.triangle_index;
    candidate.witness.patch_index = patch.patch_index;
    candidate.witness.candidate_kind = candidate_kind;
    candidate.witness.candidate_subindex = candidate_subindex;
    output = candidate;
    return true;
}

static void g1_clearance_patch_update_witness(
    G1CapsulePatchResult& result,
    const G1CapsuleWitnessCandidate& candidate)
{
    if (!result.has_witness ||
        candidate.upper < result.witness.upper ||
        (candidate.upper == result.witness.upper &&
         g1_clearance_witness_key_less(
             candidate.witness, result.witness.witness))) {
        result.has_witness = true;
        result.witness = candidate;
    }
}

static void g1_clearance_patch_update_lower(
    G1CapsulePatchResult& result,
    double lower)
{
    if (!terrain_double_is_finite(lower)) {
        return;
    }
    if (!result.has_lower || lower < result.lower) {
        result.has_lower = true;
        result.lower = lower;
    }
}

static bool g1_clearance_patch_add_coarse_lower(
    const G1CapsulePatch& patch,
    double radius,
    G1CapsulePatchResult& result)
{
    double minimum_y = patch.vertex[0].y_enclosure.lower;
    for (int index = 1; index < 3; ++index) {
        minimum_y = patch.vertex[index].y_enclosure.lower < minimum_y
            ? patch.vertex[index].y_enclosure.lower
            : minimum_y;
    }
    const volatile double coarse_value = minimum_y - radius;
    double coarse_lower = 0.0;
    if (!g1_clearance_outward_lower(
            coarse_value, coarse_lower)) {
        return false;
    }
    g1_clearance_patch_update_lower(result, coarse_lower);
    result.used_relaxed_lower = true;
    return true;
}

enum G1CapsuleFaceMembership
{
    G1CapsuleFaceInside,
    G1CapsuleFaceOutside,
    G1CapsuleFaceUncertain,
    G1CapsuleFaceDegenerate,
    G1CapsuleFaceArithmeticFailure
};

static bool g1_clearance_patch_central_barycentrics(
    const G1CapsulePatch& patch,
    double x,
    double z,
    double output[3])
{
    const volatile double dx1 =
        patch.vertex[1].x - patch.vertex[0].x;
    const volatile double dz1 =
        patch.vertex[1].z - patch.vertex[0].z;
    const volatile double dx2 =
        patch.vertex[2].x - patch.vertex[0].x;
    const volatile double dz2 =
        patch.vertex[2].z - patch.vertex[0].z;
    const volatile double px = x - patch.vertex[0].x;
    const volatile double pz = z - patch.vertex[0].z;
    const volatile double determinant = dx1 * dz2 - dx2 * dz1;
    if (!terrain_double_is_finite(determinant) || determinant == 0.0) {
        return false;
    }
    const volatile double weight1 =
        (px * dz2 - dx2 * pz) / determinant;
    const volatile double weight2 =
        (dx1 * pz - px * dz1) / determinant;
    const volatile double weight0 = 1.0 - weight1 - weight2;
    if (!terrain_double_is_finite(weight0) ||
        !terrain_double_is_finite(weight1) ||
        !terrain_double_is_finite(weight2)) {
        return false;
    }
    output[0] = weight0 == 0.0 ? 0.0 : weight0;
    output[1] = weight1 == 0.0 ? 0.0 : weight1;
    output[2] = weight2 == 0.0 ? 0.0 : weight2;
    return true;
}

static G1CapsuleFaceMembership g1_clearance_patch_face(
    const G1CertifiedEndpoint endpoints[2],
    const G1CapsuleTriangle& triangle,
    const G1CapsulePatch& patch,
    double radius,
    uint32_t primitive_index,
    G1CapsulePatchResult& result)
{
    G1ClearanceInterval dx1 = {};
    G1ClearanceInterval dz1 = {};
    G1ClearanceInterval dy1 = {};
    G1ClearanceInterval dx2 = {};
    G1ClearanceInterval dz2 = {};
    G1ClearanceInterval dy2 = {};
    if (!g1_clearance_interval_subtract(
            patch.vertex[1].x_enclosure,
            patch.vertex[0].x_enclosure, dx1) ||
        !g1_clearance_interval_subtract(
            patch.vertex[1].z_enclosure,
            patch.vertex[0].z_enclosure, dz1) ||
        !g1_clearance_interval_subtract(
            patch.vertex[1].y_enclosure,
            patch.vertex[0].y_enclosure, dy1) ||
        !g1_clearance_interval_subtract(
            patch.vertex[2].x_enclosure,
            patch.vertex[0].x_enclosure, dx2) ||
        !g1_clearance_interval_subtract(
            patch.vertex[2].z_enclosure,
            patch.vertex[0].z_enclosure, dz2) ||
        !g1_clearance_interval_subtract(
            patch.vertex[2].y_enclosure,
            patch.vertex[0].y_enclosure, dy2)) {
        return G1CapsuleFaceArithmeticFailure;
    }
    G1ClearanceInterval first_product = {};
    G1ClearanceInterval second_product = {};
    G1ClearanceInterval determinant = {};
    if (!g1_clearance_interval_multiply(dx1, dz2, first_product) ||
        !g1_clearance_interval_multiply(dx2, dz1, second_product) ||
        !g1_clearance_interval_subtract(
            first_product, second_product, determinant)) {
        return G1CapsuleFaceArithmeticFailure;
    }
    if (determinant.lower <= 0.0 && determinant.upper >= 0.0) {
        const bool exact_duplicate =
            g1_clearance_patch_vertices_same_xz(
                patch.vertex[0], patch.vertex[1]) ||
            g1_clearance_patch_vertices_same_xz(
                patch.vertex[1], patch.vertex[2]) ||
            g1_clearance_patch_vertices_same_xz(
                patch.vertex[2], patch.vertex[0]);
        if (exact_duplicate) {
            return G1CapsuleFaceDegenerate;
        }
        return g1_clearance_patch_add_coarse_lower(
                   patch, radius, result)
            ? G1CapsuleFaceUncertain
            : G1CapsuleFaceArithmeticFailure;
    }

    G1ClearanceInterval a_numerator = {};
    G1ClearanceInterval b_numerator = {};
    G1ClearanceInterval a = {};
    G1ClearanceInterval b = {};
    if (!g1_clearance_interval_multiply(dy1, dz2, first_product) ||
        !g1_clearance_interval_multiply(dy2, dz1, second_product) ||
        !g1_clearance_interval_subtract(
            first_product, second_product, a_numerator) ||
        !g1_clearance_interval_multiply(dx1, dy2, first_product) ||
        !g1_clearance_interval_multiply(dx2, dy1, second_product) ||
        !g1_clearance_interval_subtract(
            first_product, second_product, b_numerator) ||
        !g1_clearance_interval_divide(a_numerator, determinant, a) ||
        !g1_clearance_interval_divide(b_numerator, determinant, b)) {
        return G1CapsuleFaceArithmeticFailure;
    }
    G1ClearanceInterval ax0 = {};
    G1ClearanceInterval bz0 = {};
    G1ClearanceInterval c = {};
    G1ClearanceInterval partial_c = {};
    if (!g1_clearance_interval_multiply(
            a, patch.vertex[0].x_enclosure, ax0) ||
        !g1_clearance_interval_multiply(
            b, patch.vertex[0].z_enclosure, bz0) ||
        !g1_clearance_interval_subtract(
            patch.vertex[0].y_enclosure, ax0, partial_c) ||
        !g1_clearance_interval_subtract(partial_c, bz0, c)) {
        return G1CapsuleFaceArithmeticFailure;
    }
    G1ClearanceInterval a_square = {};
    G1ClearanceInterval b_square = {};
    G1ClearanceInterval n_square = {};
    G1ClearanceInterval n = {};
    if (!g1_clearance_interval_square(a, a_square) ||
        !g1_clearance_interval_square(b, b_square) ||
        !g1_clearance_interval_sum3(
            {1.0, 1.0}, a_square, b_square, n_square) ||
        !g1_clearance_interval_sqrt(n_square, n)) {
        return G1CapsuleFaceArithmeticFailure;
    }
    G1ClearanceInterval radius_n = {};
    G1ClearanceInterval stationary_value = {};
    if (!g1_clearance_interval_multiply(
            {radius, radius}, n, radius_n) ||
        !g1_clearance_interval_subtract(
            c, radius_n, stationary_value)) {
        return G1CapsuleFaceArithmeticFailure;
    }

    const volatile double dx1_c =
        patch.vertex[1].x - patch.vertex[0].x;
    const volatile double dz1_c =
        patch.vertex[1].z - patch.vertex[0].z;
    const volatile double dy1_c =
        patch.vertex[1].y - patch.vertex[0].y;
    const volatile double dx2_c =
        patch.vertex[2].x - patch.vertex[0].x;
    const volatile double dz2_c =
        patch.vertex[2].z - patch.vertex[0].z;
    const volatile double dy2_c =
        patch.vertex[2].y - patch.vertex[0].y;
    const volatile double determinant_c =
        dx1_c * dz2_c - dx2_c * dz1_c;
    if (!terrain_double_is_finite(determinant_c) ||
        determinant_c == 0.0) {
        return g1_clearance_patch_add_coarse_lower(
                   patch, radius, result)
            ? G1CapsuleFaceUncertain
            : G1CapsuleFaceArithmeticFailure;
    }
    const volatile double a_c =
        (dy1_c * dz2_c - dy2_c * dz1_c) /
        determinant_c;
    const volatile double b_c =
        (dx1_c * dy2_c - dx2_c * dy1_c) /
        determinant_c;
    const volatile double n_c =
        std::sqrt(1.0 + a_c * a_c + b_c * b_c);
    const volatile double x_c = -radius * a_c / n_c;
    const volatile double z_c = -radius * b_c / n_c;
    if (!terrain_double_is_finite(a_c) ||
        !terrain_double_is_finite(b_c) ||
        !terrain_double_is_finite(n_c) || n_c <= 0.0 ||
        !terrain_double_is_finite(x_c) ||
        !terrain_double_is_finite(z_c)) {
        return G1CapsuleFaceArithmeticFailure;
    }
    double central_weights[3] = {};
    if (!g1_clearance_patch_central_barycentrics(
            patch, x_c, z_c, central_weights)) {
        return G1CapsuleFaceArithmeticFailure;
    }

    G1ClearanceInterval negative_radius_a = {};
    G1ClearanceInterval negative_radius_b = {};
    G1ClearanceInterval x_star = {};
    G1ClearanceInterval z_star = {};
    if (!g1_clearance_interval_multiply(
            {-radius, -radius}, a, negative_radius_a) ||
        !g1_clearance_interval_multiply(
            {-radius, -radius}, b, negative_radius_b) ||
        !g1_clearance_interval_divide(
            negative_radius_a, n, x_star) ||
        !g1_clearance_interval_divide(
            negative_radius_b, n, z_star)) {
        return G1CapsuleFaceArithmeticFailure;
    }
    G1ClearanceInterval px = {};
    G1ClearanceInterval pz = {};
    G1ClearanceInterval weight1_numerator = {};
    G1ClearanceInterval weight2_numerator = {};
    G1ClearanceInterval weight1 = {};
    G1ClearanceInterval weight2 = {};
    G1ClearanceInterval weight0 = {};
    if (!g1_clearance_interval_subtract(
            x_star, patch.vertex[0].x_enclosure, px) ||
        !g1_clearance_interval_subtract(
            z_star, patch.vertex[0].z_enclosure, pz) ||
        !g1_clearance_interval_multiply(px, dz2, first_product) ||
        !g1_clearance_interval_multiply(dx2, pz, second_product) ||
        !g1_clearance_interval_subtract(
            first_product, second_product, weight1_numerator) ||
        !g1_clearance_interval_multiply(dx1, pz, first_product) ||
        !g1_clearance_interval_multiply(px, dz1, second_product) ||
        !g1_clearance_interval_subtract(
            first_product, second_product, weight2_numerator) ||
        !g1_clearance_interval_divide(
            weight1_numerator, determinant, weight1) ||
        !g1_clearance_interval_divide(
            weight2_numerator, determinant, weight2) ||
        !g1_clearance_interval_subtract(
            {1.0, 1.0}, weight1, partial_c) ||
        !g1_clearance_interval_subtract(
            partial_c, weight2, weight0)) {
        return G1CapsuleFaceArithmeticFailure;
    }
    const G1ClearanceInterval weights[3] = {
        weight0, weight1, weight2
    };
    bool certified_inside = true;
    bool certified_outside = false;
    for (int index = 0; index < 3; ++index) {
        certified_inside = certified_inside &&
            weights[index].lower >= 0.0;
        certified_outside = certified_outside ||
            weights[index].upper < 0.0;
    }
    if (!certified_outside) {
        g1_clearance_patch_update_lower(
            result, stationary_value.lower);
        bool central_inside = true;
        for (int index = 0; index < 3; ++index) {
            central_inside = central_inside &&
                central_weights[index] >= 0.0;
        }
        if (central_inside) {
            G1CapsuleWitnessCandidate witness = {};
            if (g1_clearance_patch_witness(
                    endpoints, triangle, patch, central_weights,
                    radius, primitive_index, 0, 0, witness)) {
                g1_clearance_patch_update_witness(result, witness);
            }
        }
    }
    if (certified_inside) {
        return G1CapsuleFaceInside;
    }
    if (certified_outside) {
        return G1CapsuleFaceOutside;
    }
    result.used_relaxed_lower = true;
    return G1CapsuleFaceUncertain;
}

static bool g1_clearance_line_value(
    const G1ClearanceInterval& c,
    const G1ClearanceInterval& m,
    const G1ClearanceInterval& radius_square,
    const G1ClearanceInterval& s,
    G1ClearanceInterval& output)
{
    G1ClearanceInterval ms = {};
    G1ClearanceInterval affine = {};
    G1ClearanceInterval s_square = {};
    G1ClearanceInterval radicand = {};
    G1ClearanceInterval root = {};
    if (!g1_clearance_interval_multiply(m, s, ms) ||
        !g1_clearance_interval_add(c, ms, affine) ||
        !g1_clearance_interval_square(s, s_square) ||
        !g1_clearance_interval_subtract(
            radius_square, s_square, radicand) ||
        radicand.upper < 0.0) {
        return false;
    }
    radicand.lower = radicand.lower < 0.0
        ? 0.0
        : radicand.lower;
    return g1_clearance_interval_sqrt(radicand, root) &&
           g1_clearance_interval_subtract(affine, root, output);
}

static bool g1_clearance_patch_edge(
    const G1CertifiedEndpoint endpoints[2],
    const G1CapsuleTriangle& triangle,
    const G1CapsulePatch& patch,
    uint32_t first_index,
    uint32_t second_index,
    double radius,
    uint32_t primitive_index,
    G1CapsulePatchResult& result)
{
    const G1CapsulePatchVertex& first = patch.vertex[first_index];
    const G1CapsulePatchVertex& second = patch.vertex[second_index];
    G1ClearanceInterval dx = {};
    G1ClearanceInterval dz = {};
    G1ClearanceInterval dy = {};
    G1ClearanceInterval dx_square = {};
    G1ClearanceInterval dz_square = {};
    G1ClearanceInterval length_square = {};
    G1ClearanceInterval radius_square = {};
    if (!g1_clearance_interval_subtract(
            second.x_enclosure, first.x_enclosure, dx) ||
        !g1_clearance_interval_subtract(
            second.z_enclosure, first.z_enclosure, dz) ||
        !g1_clearance_interval_subtract(
            second.y_enclosure, first.y_enclosure, dy) ||
        !g1_clearance_interval_square(dx, dx_square) ||
        !g1_clearance_interval_square(dz, dz_square) ||
        !g1_clearance_interval_add(
            dx_square, dz_square, length_square) ||
        !g1_clearance_interval_square(
            {radius, radius}, radius_square)) {
        return false;
    }
    const volatile double dx_c = second.x - first.x;
    const volatile double dz_c = second.z - first.z;
    const volatile double dy_c = second.y - first.y;
    const volatile double length_square_c =
        dx_c * dx_c + dz_c * dz_c;
    if (!terrain_double_is_finite(length_square_c) ||
        length_square_c < 0.0) {
        return false;
    }
    const uint32_t edge_index = first_index;
    if (length_square_c == 0.0) {
        if (!g1_clearance_patch_vertices_same_xz(first, second)) {
            return g1_clearance_patch_add_coarse_lower(
                patch, radius, result);
        }
        G1ClearanceInterval x_square = {};
        G1ClearanceInterval z_square = {};
        G1ClearanceInterval rho_square = {};
        G1ClearanceInterval radicand = {};
        G1ClearanceInterval root = {};
        G1ClearanceInterval minimum_y = {};
        G1ClearanceInterval value = {};
        if (!g1_clearance_interval_square(
                first.x_enclosure, x_square) ||
            !g1_clearance_interval_square(
                first.z_enclosure, z_square) ||
            !g1_clearance_interval_add(
                x_square, z_square, rho_square)) {
            return false;
        }
        if (rho_square.lower > radius_square.upper) {
            return true;
        }
        if (!g1_clearance_interval_subtract(
                radius_square, rho_square, radicand) ||
            radicand.upper < 0.0) {
            return true;
        }
        radicand.lower = radicand.lower < 0.0
            ? 0.0
            : radicand.lower;
        if (!g1_clearance_interval_sqrt(radicand, root) ||
            !g1_clearance_interval_minimum(
                first.y_enclosure, second.y_enclosure, minimum_y) ||
            !g1_clearance_interval_subtract(
                minimum_y, root, value)) {
            return false;
        }
        g1_clearance_patch_update_lower(result, value.lower);
        const uint32_t selected = first.y <= second.y
            ? first_index
            : second_index;
        double weights[3] = {};
        weights[selected] = 1.0;
        G1CapsuleWitnessCandidate witness = {};
        if (g1_clearance_patch_witness(
                endpoints, triangle, patch, weights,
                radius, primitive_index, 1,
                3 * edge_index + 1, witness)) {
            g1_clearance_patch_update_witness(result, witness);
        }
        if (rho_square.upper > radius_square.lower) {
            result.used_relaxed_lower = true;
        }
        return true;
    }
    if (length_square.lower <= 0.0) {
        return g1_clearance_patch_add_coarse_lower(
            patch, radius, result);
    }
    G1ClearanceInterval length = {};
    G1ClearanceInterval cross_first = {};
    G1ClearanceInterval cross_second = {};
    G1ClearanceInterval cross = {};
    G1ClearanceInterval cross_square = {};
    G1ClearanceInterval q_square = {};
    if (!g1_clearance_interval_sqrt(length_square, length) ||
        !g1_clearance_interval_multiply(
            first.x_enclosure, dz, cross_first) ||
        !g1_clearance_interval_multiply(
            first.z_enclosure, dx, cross_second) ||
        !g1_clearance_interval_subtract(
            cross_first, cross_second, cross) ||
        !g1_clearance_interval_square(cross, cross_square) ||
        !g1_clearance_interval_divide(
            cross_square, length_square, q_square)) {
        return false;
    }
    if (q_square.lower > radius_square.upper) {
        return true;
    }
    G1ClearanceInterval line_radius_square = {};
    if (!g1_clearance_interval_subtract(
            radius_square, q_square, line_radius_square) ||
        line_radius_square.upper < 0.0) {
        return true;
    }
    line_radius_square.lower = line_radius_square.lower < 0.0
        ? 0.0
        : line_radius_square.lower;
    G1ClearanceInterval line_radius = {};
    G1ClearanceInterval dot_first = {};
    G1ClearanceInterval dot_second = {};
    G1ClearanceInterval dot = {};
    G1ClearanceInterval s0 = {};
    G1ClearanceInterval s1 = {};
    G1ClearanceInterval m = {};
    G1ClearanceInterval ms0 = {};
    G1ClearanceInterval c = {};
    if (!g1_clearance_interval_sqrt(
            line_radius_square, line_radius) ||
        !g1_clearance_interval_multiply(
            first.x_enclosure, dx, dot_first) ||
        !g1_clearance_interval_multiply(
            first.z_enclosure, dz, dot_second) ||
        !g1_clearance_interval_add(dot_first, dot_second, dot) ||
        !g1_clearance_interval_divide(dot, length, s0) ||
        !g1_clearance_interval_add(s0, length, s1) ||
        !g1_clearance_interval_divide(dy, length, m) ||
        !g1_clearance_interval_multiply(m, s0, ms0) ||
        !g1_clearance_interval_subtract(
            first.y_enclosure, ms0, c)) {
        return false;
    }
    G1ClearanceInterval negative_radius = {};
    if (!g1_clearance_interval_negate(
            line_radius, negative_radius)) {
        return false;
    }
    G1ClearanceInterval feasible_lower = {};
    G1ClearanceInterval feasible_upper = {};
    if (!g1_clearance_interval_maximum(
            s0, negative_radius, feasible_lower) ||
        !g1_clearance_interval_minimum(
            s1, line_radius, feasible_upper)) {
        return false;
    }
    if (feasible_lower.lower > feasible_upper.upper) {
        return true;
    }
    G1ClearanceInterval m_square = {};
    G1ClearanceInterval line_n_square = {};
    G1ClearanceInterval line_n = {};
    G1ClearanceInterval radius_n = {};
    G1ClearanceInterval unconstrained = {};
    G1ClearanceInterval negative_m = {};
    G1ClearanceInterval stationary_numerator = {};
    G1ClearanceInterval stationary = {};
    if (!g1_clearance_interval_square(m, m_square) ||
        !g1_clearance_interval_add(
            {1.0, 1.0}, m_square, line_n_square) ||
        !g1_clearance_interval_sqrt(line_n_square, line_n) ||
        !g1_clearance_interval_multiply(
            line_radius, line_n, radius_n) ||
        !g1_clearance_interval_subtract(c, radius_n, unconstrained) ||
        !g1_clearance_interval_negate(m, negative_m) ||
        !g1_clearance_interval_multiply(
            negative_m, line_radius, stationary_numerator) ||
        !g1_clearance_interval_divide(
            stationary_numerator, line_n, stationary)) {
        return false;
    }

    double selected_lower = unconstrained.lower;
    bool exact_branch = false;
    if (feasible_lower.upper <= feasible_upper.lower) {
        if (stationary.lower >= feasible_lower.upper &&
            stationary.upper <= feasible_upper.lower) {
            exact_branch = true;
        } else if (stationary.upper < feasible_lower.lower) {
            G1ClearanceInterval endpoint_value = {};
            if (!g1_clearance_line_value(
                    c, m, line_radius_square,
                    feasible_lower, endpoint_value)) {
                return false;
            }
            selected_lower = endpoint_value.lower;
            exact_branch = true;
        } else if (stationary.lower > feasible_upper.upper) {
            G1ClearanceInterval endpoint_value = {};
            if (!g1_clearance_line_value(
                    c, m, line_radius_square,
                    feasible_upper, endpoint_value)) {
                return false;
            }
            selected_lower = endpoint_value.lower;
            exact_branch = true;
        }
    }
    if (!exact_branch) {
        result.used_relaxed_lower = true;
    }
    g1_clearance_patch_update_lower(result, selected_lower);

    const volatile double length_c = std::sqrt(length_square_c);
    const volatile double cross_c = first.x * dz_c - first.z * dx_c;
    const volatile double q_square_c =
        (cross_c * cross_c) / length_square_c;
    const volatile double line_radius_square_c =
        radius * radius - q_square_c;
    if (!terrain_double_is_finite(length_c) || length_c <= 0.0 ||
        !terrain_double_is_finite(line_radius_square_c) ||
        line_radius_square_c < 0.0) {
        return g1_clearance_patch_add_coarse_lower(
            patch, radius, result);
    }
    const volatile double line_radius_c =
        std::sqrt(line_radius_square_c);
    const volatile double s0_c =
        (first.x * dx_c + first.z * dz_c) / length_c;
    const volatile double s1_c = s0_c + length_c;
    const volatile double m_c = dy_c / length_c;
    const volatile double stationary_c =
        -m_c * line_radius_c /
        std::sqrt(1.0 + m_c * m_c);
    const double lower_c = s0_c > -line_radius_c
        ? s0_c
        : -line_radius_c;
    const double upper_c = s1_c < line_radius_c
        ? s1_c
        : line_radius_c;
    if (!terrain_double_is_finite(lower_c) ||
        !terrain_double_is_finite(upper_c) || lower_c > upper_c) {
        return g1_clearance_patch_add_coarse_lower(
            patch, radius, result);
    }
    const double candidate_s[3] = {
        stationary_c < lower_c
            ? lower_c
            : (stationary_c > upper_c ? upper_c : stationary_c),
        lower_c,
        upper_c
    };
    for (uint32_t local = 0; local < 3; ++local) {
        if (local == 0 &&
            (stationary_c < lower_c || stationary_c > upper_c)) {
            continue;
        }
        const volatile double edge_parameter =
            (candidate_s[local] - s0_c) / length_c;
        if (!terrain_double_is_finite(edge_parameter) ||
            edge_parameter < 0.0 || edge_parameter > 1.0) {
            continue;
        }
        double weights[3] = {};
        weights[first_index] = 1.0 - edge_parameter;
        weights[second_index] = edge_parameter;
        G1CapsuleWitnessCandidate witness = {};
        if (g1_clearance_patch_witness(
                endpoints, triangle, patch, weights,
                radius, primitive_index, 1,
                3 * edge_index + local, witness)) {
            g1_clearance_patch_update_witness(result, witness);
        }
    }
    return true;
}

static bool g1_clearance_solve_capsule_patch(
    const G1CertifiedEndpoint endpoints[2],
    const G1CapsuleTriangle& triangle,
    const G1CapsulePatch& patch,
    double radius,
    uint32_t primitive_index,
    G1CapsulePatchResult& output)
{
    G1CapsulePatchResult candidate = {};
    const G1CapsuleFaceMembership face = g1_clearance_patch_face(
        endpoints, triangle, patch, radius,
        primitive_index, candidate);
    if (face == G1CapsuleFaceArithmeticFailure) {
        return false;
    }
    const uint32_t edges[3][2] = {
        {0, 1}, {1, 2}, {2, 0}
    };
    for (uint32_t edge = 0; edge < 3; ++edge) {
        if (!g1_clearance_patch_edge(
                endpoints, triangle, patch,
                edges[edge][0], edges[edge][1],
                radius, primitive_index, candidate)) {
            return false;
        }
    }
    if (face == G1CapsuleFaceOutside && !candidate.has_lower) {
        output = candidate;
        return true;
    }
    if (face == G1CapsuleFaceDegenerate) {
        // A rank-deficient projected triangle has every objective fiber
        // represented on its source boundary.  The three exact edges cover
        // its minimum; any uncertain edge has already marked the relaxed
        // result for bounded fallback.
    }
    output = candidate;
    return true;
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

static G1ClearanceStatus g1_clearance_capsule_core(
    G1ClearanceResult& output,
    const G1ClearanceBudget& limits,
    const heightfield& field,
    vec3 endpoint_a,
    vec3 endpoint_b,
    float radius_m,
    const G1ClearanceDiagnostic& diagnostic)
{
    if (field.version != 2 ||
        !terrain_heightfield_is_queryable(field)) {
        return g1_clearance_error(
            G1ClearanceInvalidField,
            diagnostic.output, diagnostic.capacity,
            "G1 capsule clearance requires a queryable G1HF/v2 field");
    }
    if (!g1_ik_vec3_is_runtime_value(endpoint_a) ||
        !g1_ik_vec3_is_runtime_value(endpoint_b) ||
        !terrain_float_is_normal_or_positive_zero(radius_m) ||
        radius_m <= 0.0f) {
        return g1_clearance_error(
            G1ClearanceInvalidInput,
            diagnostic.output, diagnostic.capacity,
            "G1 capsule requires runtime endpoints and positive-normal radius");
    }
    endpoint_a = vec3(
        terrain_runtime_canonicalize_output(endpoint_a.x),
        terrain_runtime_canonicalize_output(endpoint_a.y),
        terrain_runtime_canonicalize_output(endpoint_a.z));
    endpoint_b = vec3(
        terrain_runtime_canonicalize_output(endpoint_b.x),
        terrain_runtime_canonicalize_output(endpoint_b.y),
        terrain_runtime_canonicalize_output(endpoint_b.z));
    const double radius = static_cast<double>(radius_m);
    G1CertifiedEndpoint endpoints[2] = {};
    if (!g1_clearance_make_public_endpoints(
            endpoint_a, endpoint_b, endpoints) ||
        g1_clearance_endpoint_key_less(
            endpoints[1].source_key,
            endpoints[0].source_key)) {
        return g1_clearance_error(
            G1ClearanceArithmeticFailure,
            diagnostic.output, diagnostic.capacity,
            "G1 capsule endpoints could not be canonicalized");
    }

    G1ClearanceCellSpan span = {};
    const G1ClearanceDomainStatus domain_status =
        g1_clearance_footprint_span(
            field, endpoints, radius, span);
    if (domain_status != G1ClearanceDomainInside) {
        return g1_clearance_error(
            domain_status == G1ClearanceDomainOutside
                ? G1ClearanceOutsideDomain
                : G1ClearanceArithmeticFailure,
            diagnostic.output, diagnostic.capacity,
            domain_status == G1ClearanceDomainOutside
                ? "G1 capsule footprint is outside the G1HF/v2 domain"
                : "G1 capsule footprint bounds could not be certified");
    }

    uint64_t pair_count = 0;
    uint64_t patch_count = 0;
    uint64_t candidate_count = 0;
    if (!g1_clearance_checked_multiply_u64(
            span.cell_count, UINT64_C(2), pair_count) ||
        !g1_clearance_checked_multiply_u64(
            pair_count,
            static_cast<uint64_t>(G1ClearancePatchesPerPair),
            patch_count) ||
        !g1_clearance_checked_multiply_u64(
            pair_count,
            static_cast<uint64_t>(G1ClearanceCandidatesPerPair),
            candidate_count)) {
        return g1_clearance_error(
            G1ClearanceArithmeticFailure,
            diagnostic.output, diagnostic.capacity,
            "G1 capsule checked work products overflowed");
    }
    if (span.cell_count > G1ClearanceMaximumCellsPerPrimitive ||
        pair_count > G1ClearanceMaximumPairsPerPrimitive ||
        span.cell_count > limits.maximum_cells ||
        pair_count > limits.maximum_primitive_triangle_pairs ||
        patch_count > limits.maximum_face_patches ||
        candidate_count > limits.maximum_candidate_tests) {
        return g1_clearance_error(
            G1ClearanceBudgetExceeded,
            diagnostic.output, diagnostic.capacity,
            "G1 capsule fixed work exceeds its tightened budget");
    }

    bool has_lower = false;
    double global_lower = 0.0;
    bool has_witness = false;
    G1CapsuleWitnessCandidate global_witness = {};
    bool relaxed_patch_remains = false;
    for (int cell_z = span.minimum_z;
         cell_z <= span.maximum_z;
         ++cell_z) {
        for (int cell_x = span.minimum_x;
             cell_x <= span.maximum_x;
             ++cell_x) {
            heightfield_cell cell = {};
            cell.x0 = cell_x;
            cell.z0 = cell_z;
            double h00 = 0.0;
            double h10 = 0.0;
            double h01 = 0.0;
            double h11 = 0.0;
            if (!terrain_v2_cell_heights(
                    field, cell, h00, h10, h01, h11)) {
                return g1_clearance_error(
                    G1ClearanceInvalidField,
                    diagnostic.output, diagnostic.capacity,
                    "G1 capsule encountered invalid visited heights");
            }
            for (uint32_t triangle_index = 0;
                 triangle_index < 2;
                 ++triangle_index) {
                G1CapsuleTriangle triangle = {};
                if (!g1_clearance_make_capsule_triangle(
                        field, cell_x, cell_z, triangle_index,
                        h00, h10, h01, h11, triangle)) {
                    return g1_clearance_error(
                        G1ClearanceArithmeticFailure,
                        diagnostic.output, diagnostic.capacity,
                        "G1 capsule source triangle could not be constructed");
                }
                G1PointTriangle guard_triangle = {};
                for (int vertex = 0; vertex < 3; ++vertex) {
                    guard_triangle.x[vertex] = triangle.x[vertex];
                    guard_triangle.z[vertex] = triangle.z[vertex];
                    guard_triangle.height[vertex] =
                        triangle.height[vertex];
                }
                guard_triangle.index = triangle_index;
                double float_guard = 0.0;
                if (!g1_clearance_point_float_guard(
                        guard_triangle, float_guard)) {
                    return g1_clearance_error(
                        G1ClearanceArithmeticFailure,
                        diagnostic.output, diagnostic.capacity,
                        "G1 capsule output guard could not be derived");
                }
                if (!terrain_double_is_finite(float_guard) ||
                    float_guard >
                        G1ClearanceMaximumCertificateWidthM) {
                    return g1_clearance_error(
                        G1ClearanceUncertified,
                        diagnostic.output, diagnostic.capacity,
                        "G1 capsule mandatory float-output guard is too wide");
                }
                for (uint32_t patch_index = 0;
                     patch_index < G1ClearancePatchesPerPair;
                     ++patch_index) {
                    G1CapsulePatch patch = {};
                    G1CapsulePatchResult patch_result = {};
                    if (!g1_clearance_make_capsule_patch(
                            endpoints, triangle,
                            patch_index, patch) ||
                        !g1_clearance_solve_capsule_patch(
                            endpoints, triangle, patch, radius,
                            0, patch_result)) {
                        return g1_clearance_error(
                            G1ClearanceArithmeticFailure,
                            diagnostic.output, diagnostic.capacity,
                            "G1 capsule patch arithmetic failed");
                    }
                    if (patch_result.has_lower) {
                        const volatile double guarded_value =
                            patch_result.lower - float_guard;
                        double guarded_lower = 0.0;
                        if (!g1_clearance_outward_lower(
                                guarded_value, guarded_lower)) {
                            return g1_clearance_error(
                                G1ClearanceArithmeticFailure,
                                diagnostic.output, diagnostic.capacity,
                                "G1 capsule guarded lower bound is nonfinite");
                        }
                        if (!has_lower || guarded_lower < global_lower) {
                            has_lower = true;
                            global_lower = guarded_lower;
                        }
                    }
                    if (patch_result.has_witness &&
                        (!has_witness ||
                         patch_result.witness.upper <
                             global_witness.upper ||
                         (patch_result.witness.upper ==
                              global_witness.upper &&
                          g1_clearance_witness_key_less(
                              patch_result.witness.witness,
                              global_witness.witness)))) {
                        has_witness = true;
                        global_witness = patch_result.witness;
                    }
                    relaxed_patch_remains = relaxed_patch_remains ||
                        patch_result.used_relaxed_lower;
                }
            }
        }
    }
    if (!has_lower || !has_witness ||
        !terrain_double_is_finite(global_lower) ||
        !terrain_double_is_finite(global_witness.upper) ||
        global_lower > global_witness.upper) {
        return g1_clearance_error(
            G1ClearanceUncertified,
            diagnostic.output, diagnostic.capacity,
            "G1 capsule did not obtain both certified endpoints");
    }
    const volatile double width_value =
        global_witness.upper - global_lower;
    double width_upper = 0.0;
    if (!g1_clearance_outward_upper(
            width_value, width_upper)) {
        return g1_clearance_error(
            G1ClearanceArithmeticFailure,
            diagnostic.output, diagnostic.capacity,
            "G1 capsule certificate width is nonfinite");
    }
    if (width_upper > G1ClearanceMaximumCertificateWidthM) {
        return g1_clearance_error(
            G1ClearanceUncertified,
            diagnostic.output, diagnostic.capacity,
            relaxed_patch_remains
                ? "G1 capsule relaxed patch requires bounded fallback"
                : "G1 capsule analytic certificate is too wide");
    }

    G1ClearanceResult candidate = {};
    candidate.lower_bound_m = global_lower;
    candidate.witness_upper_m = global_witness.upper;
    candidate.witness = global_witness.witness;
    candidate.work.cells_visited =
        static_cast<uint32_t>(span.cell_count);
    candidate.work.primitive_triangle_pairs =
        static_cast<uint32_t>(pair_count);
    candidate.work.face_patches =
        static_cast<uint32_t>(patch_count);
    candidate.work.candidate_tests =
        static_cast<uint32_t>(candidate_count);
    output = candidate;
    return G1ClearanceOk;
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
    const heightfield& field,
    vec3 center,
    float radius_m,
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
            "G1 sphere budget exceeds the pose factory ceiling");
    }
    return g1_clearance_capsule_core(
        output, limits, field,
        center, center, radius_m, diagnostic);
}

G1ClearanceStatus g1_capsule_clearance(
    G1ClearanceResult& output,
    const G1ClearanceBudget& limits,
    const heightfield& field,
    vec3 endpoint_a,
    vec3 endpoint_b,
    float radius_m,
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
            "G1 capsule budget exceeds the pose factory ceiling");
    }
    return g1_clearance_capsule_core(
        output, limits, field,
        endpoint_a, endpoint_b, radius_m, diagnostic);
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
