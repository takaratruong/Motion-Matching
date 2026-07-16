#if defined(__FAST_MATH__)
#error "G1 root-reach kernel must be compiled without fast math"
#endif

#include "g1_ik.h"

bool ik_effective_reach_shell(
    IKReachShell& output,
    const vec3& root,
    const vec3& middle,
    const vec3& end,
    float reach_buffer_m)
{
    if (g1_ik_memory_ranges_overlap(
            &output, sizeof(output), &root, sizeof(root)) ||
        g1_ik_memory_ranges_overlap(
            &output, sizeof(output), &middle, sizeof(middle)) ||
        g1_ik_memory_ranges_overlap(
            &output, sizeof(output), &end, sizeof(end)) ||
        !ik_vec3_is_runtime_value(root) ||
        !ik_vec3_is_runtime_value(middle) ||
        !ik_vec3_is_runtime_value(end) ||
        !terrain_float_is_positive_normal(reach_buffer_m)) {
        return false;
    }

    double upper = 0.0;
    double lower = 0.0;
    double current = 0.0;
    float upper_f32 = 0.0f;
    float lower_f32 = 0.0f;
    float current_f32 = 0.0f;
    if (!ik_checked_distance_precise(
            upper, upper_f32, middle, root) ||
        !ik_checked_distance_precise(
            lower, lower_f32, end, middle) ||
        !ik_checked_distance_precise(
            current, current_f32, end, root) ||
        upper <= static_cast<double>(reach_buffer_m) ||
        lower <= static_cast<double>(reach_buffer_m)) {
        return false;
    }

    const volatile double nominal_minimum =
        std::fabs(upper - lower) +
        static_cast<double>(reach_buffer_m);
    const volatile double nominal_maximum =
        upper + lower - static_cast<double>(reach_buffer_m);
    if (!terrain_double_is_finite(nominal_minimum) ||
        !terrain_double_is_finite(nominal_maximum) ||
        nominal_minimum <= 0.0 ||
        nominal_maximum < nominal_minimum) {
        return false;
    }

    IKReachShell candidate = {};
    candidate.minimum_distance_m =
        current < nominal_minimum ? current : nominal_minimum;
    candidate.maximum_distance_m =
        current > nominal_maximum ? current : nominal_maximum;
    if (!ik_checked_binary32_commit(
            candidate.minimum_distance_m,
            candidate.minimum_distance_f32_m) ||
        !ik_checked_binary32_commit(
            candidate.maximum_distance_m,
            candidate.maximum_distance_f32_m)) {
        return false;
    }
    output = candidate;
    return true;
}

bool ik_project_target(
    IKTargetProjection& output,
    vec3 root,
    vec3 middle,
    vec3 end,
    vec3 requested,
    float reach_buffer_m)
{
    if (!ik_vec3_is_runtime_value(root) ||
        !ik_vec3_is_runtime_value(middle) ||
        !ik_vec3_is_runtime_value(end) ||
        !ik_vec3_is_runtime_value(requested) ||
        !terrain_float_is_positive_normal(reach_buffer_m)) {
        return false;
    }

    IKReachShell shell = {};
    double current_distance = 0.0;
    double raw_distance = 0.0;
    float current_float = 0.0f;
    float raw_float = 0.0f;
    if (!ik_effective_reach_shell(
            shell, root, middle, end, reach_buffer_m) ||
        !ik_checked_distance_precise(
            current_distance, current_float, end, root) ||
        !ik_checked_distance_precise(
            raw_distance, raw_float, requested, root)) {
        return false;
    }
    const double effective_minimum = shell.minimum_distance_m;
    const double effective_maximum = shell.maximum_distance_m;

    vec3 direction(0.0f, -1.0f, 0.0f);
    if (raw_distance > 1.0e-7) {
        if (!ik_checked_vec3_from_double(
                direction,
                (static_cast<double>(requested.x) -
                    static_cast<double>(root.x)) / raw_distance,
                (static_cast<double>(requested.y) -
                    static_cast<double>(root.y)) / raw_distance,
                (static_cast<double>(requested.z) -
                    static_cast<double>(root.z)) / raw_distance) ||
            !ik_vec3_is_unit(direction)) {
            return false;
        }
    } else if (current_distance > 1.0e-7) {
        if (!ik_checked_vec3_from_double(
                direction,
                (static_cast<double>(end.x) -
                    static_cast<double>(root.x)) / current_distance,
                (static_cast<double>(end.y) -
                    static_cast<double>(root.y)) / current_distance,
                (static_cast<double>(end.z) -
                    static_cast<double>(root.z)) / current_distance) ||
            !ik_vec3_is_unit(direction)) {
            return false;
        }
    }

    vec3 clamped_target;
    double materialized_distance = 0.0;
    float clamped_float = 0.0f;
    const bool reachable =
        raw_distance >= effective_minimum &&
        raw_distance <= effective_maximum;
    if (reachable) {
        clamped_target = requested;
        materialized_distance = raw_distance;
        clamped_float = raw_float;
    } else {
        const double boundary_distance =
            raw_distance < effective_minimum
                ? effective_minimum
                : effective_maximum;
        if (!ik_checked_materialize_shell_target(
                clamped_target,
                materialized_distance,
                clamped_float,
                root, direction, boundary_distance,
                effective_minimum, effective_maximum)) {
            return false;
        }
    }
    if (materialized_distance < effective_minimum ||
        materialized_distance > effective_maximum) {
        return false;
    }

    IKTargetProjection candidate = {};
    candidate.reachable = reachable;
    candidate.clamped_target = clamped_target;
    candidate.raw_distance_m = raw_float;
    candidate.clamped_distance_m = clamped_float;
    candidate.minimum_distance_m = shell.minimum_distance_f32_m;
    candidate.maximum_distance_m = shell.maximum_distance_f32_m;
    output = candidate;
    return true;
}

bool g1_physical_sole_position_target(
    G1PhysicalSolePositionTarget& output,
    const vec3& current_contact_origin,
    const quat& current_contact_rotation,
    const vec3& current_ankle_origin,
    const G1LegConfig& config,
    const vec3& desired_sole_center,
    const vec3& desired_sole_normal,
    char* error,
    int error_capacity)
{
    const std::size_t error_bytes =
        error != NULL && error_capacity > 0
            ? static_cast<std::size_t>(error_capacity)
            : 0U;
    const bool output_alias =
        g1_ik_memory_ranges_overlap(
            &output, sizeof(output),
            &current_contact_origin, sizeof(current_contact_origin)) ||
        g1_ik_memory_ranges_overlap(
            &output, sizeof(output),
            &current_contact_rotation, sizeof(current_contact_rotation)) ||
        g1_ik_memory_ranges_overlap(
            &output, sizeof(output),
            &current_ankle_origin, sizeof(current_ankle_origin)) ||
        g1_ik_memory_ranges_overlap(
            &output, sizeof(output), &config, sizeof(config)) ||
        g1_ik_memory_ranges_overlap(
            &output, sizeof(output),
            &desired_sole_center, sizeof(desired_sole_center)) ||
        g1_ik_memory_ranges_overlap(
            &output, sizeof(output),
            &desired_sole_normal, sizeof(desired_sole_normal));
    const bool error_alias =
        g1_ik_memory_ranges_overlap(
            error, error_bytes, &output, sizeof(output)) ||
        g1_ik_memory_ranges_overlap(
            error, error_bytes,
            &current_contact_origin, sizeof(current_contact_origin)) ||
        g1_ik_memory_ranges_overlap(
            error, error_bytes,
            &current_contact_rotation, sizeof(current_contact_rotation)) ||
        g1_ik_memory_ranges_overlap(
            error, error_bytes,
            &current_ankle_origin, sizeof(current_ankle_origin)) ||
        g1_ik_memory_ranges_overlap(
            error, error_bytes, &config, sizeof(config)) ||
        g1_ik_memory_ranges_overlap(
            error, error_bytes,
            &desired_sole_center, sizeof(desired_sole_center)) ||
        g1_ik_memory_ranges_overlap(
            error, error_bytes,
            &desired_sole_normal, sizeof(desired_sole_normal));
    if (error_capacity < 0 || output_alias || error_alias ||
        !g1_foot_runtime_config_validate(
            config, error, error_capacity) ||
        !g1_ik_vec3_is_runtime_value(current_contact_origin) ||
        !ik_quat_is_unit(current_contact_rotation) ||
        !g1_ik_vec3_is_runtime_value(current_ankle_origin) ||
        !g1_ik_vec3_is_runtime_value(desired_sole_center) ||
        !g1_ik_surface_normal_is_valid(desired_sole_normal)) {
        return false;
    }

    G1PhysicalSolePositionTarget target = {};
    vec3 sole_offset;
    vec3 contact_to_ankle;
    if (!g1_surface_aligned_foot_rotation(
            target.contact_rotation,
            current_contact_rotation,
            config,
            desired_sole_normal,
            error,
            error_capacity) ||
        !g1_ik_checked_physical_sole_centroid(
            sole_offset,
            vec3(),
            target.contact_rotation,
            config) ||
        !ik_checked_vec3_subtract(
            target.contact_origin,
            desired_sole_center,
            sole_offset) ||
        !ik_checked_vec3_subtract(
            contact_to_ankle,
            current_contact_origin,
            current_ankle_origin) ||
        !ik_checked_vec3_subtract(
            target.ankle_target,
            target.contact_origin,
            contact_to_ankle) ||
        !ik_quat_is_unit(target.contact_rotation) ||
        !g1_ik_vec3_is_runtime_value(target.contact_origin) ||
        !g1_ik_vec3_is_runtime_value(target.ankle_target)) {
        return g1_ik_error(
            error,
            error_capacity,
            "G1 physical sole target arithmetic failed");
    }
    output = target;
    return true;
}

namespace
{

struct G1RootReachInterval
{
    double lower_m;
    double upper_m;
};

struct G1RootReachIntervalSet
{
    G1RootReachInterval values[4];
    uint32_t count;
};

struct G1RootReachFootGeometry
{
    bool active;
    vec3 hip;
    vec3 knee;
    vec3 ankle;
    vec3 baseline_ankle_target;
    vec3 desired_sole_center;
    vec3 desired_sole_normal;
    IKReachShell shell;
};

static constexpr uint32_t
    G1RootReachMaximumBoundaryCandidateTests = 32U;

struct G1RootReachCandidateCursor
{
    bool active;
    uint32_t interval_index;
    G1RootReachInterval interval;
    float delta_m;
};

enum G1RootReachCandidateStatus
{
    G1RootReachCandidateInvalid = 0,
    G1RootReachCandidateRejected,
    G1RootReachCandidateAccepted,
};

static bool g1_root_reach_interval_is_valid(
    const G1RootReachInterval& interval)
{
    return terrain_double_is_finite(interval.lower_m) &&
           terrain_double_is_finite(interval.upper_m) &&
           interval.lower_m <= interval.upper_m;
}

static bool g1_root_reach_checked_add(
    double left,
    double right,
    double& output)
{
    if (!terrain_double_is_finite(left) ||
        !terrain_double_is_finite(right)) {
        return false;
    }
    const volatile double sum = left + right;
    if (!terrain_double_is_finite(sum)) return false;
    output = sum;
    return true;
}

static bool g1_root_reach_checked_subtract(
    double left,
    double right,
    double& output)
{
    if (!terrain_double_is_finite(left) ||
        !terrain_double_is_finite(right)) {
        return false;
    }
    const volatile double difference = left - right;
    if (!terrain_double_is_finite(difference)) return false;
    output = difference;
    return true;
}

static bool g1_root_reach_checked_multiply(
    double left,
    double right,
    double& output)
{
    if (!terrain_double_is_finite(left) ||
        !terrain_double_is_finite(right)) {
        return false;
    }
    const volatile double product = left * right;
    if (!terrain_double_is_finite(product)) return false;
    output = product;
    return true;
}

static bool g1_root_reach_checked_forward_kinematics(
    vec3 (&global_positions)[G1_BoneCount],
    quat (&global_rotations)[G1_BoneCount],
    const slice1d<vec3> local_positions,
    const slice1d<quat> local_rotations,
    const slice1d<int> parents)
{
    for (int bone = 0; bone < G1_BoneCount; ++bone) {
        if (!g1_ik_vec3_is_runtime_value(local_positions.data[bone]) ||
            !ik_quat_is_unit(local_rotations.data[bone])) {
            return false;
        }
        const int parent = parents.data[bone];
        if (parent == -1) {
            global_positions[bone] = local_positions.data[bone];
            global_rotations[bone] = local_rotations.data[bone];
        } else {
            vec3 offset;
            if (!ik_checked_quat_rotate(
                    offset,
                    global_rotations[parent],
                    local_positions.data[bone]) ||
                !ik_checked_vec3_add(
                    global_positions[bone],
                    global_positions[parent],
                    offset) ||
                !ik_checked_quat_multiply(
                    global_rotations[bone],
                    global_rotations[parent],
                    local_rotations.data[bone]) ||
                !g1_ik_vec3_is_runtime_value(global_positions[bone]) ||
                !ik_quat_is_unit(global_rotations[bone])) {
                return false;
            }
        }
    }
    return true;
}

static bool g1_root_reach_interval_set_sort(
    G1RootReachIntervalSet& intervals)
{
    if (intervals.count > 4U) return false;
    for (uint32_t index = 0U; index < intervals.count; ++index) {
        if (!g1_root_reach_interval_is_valid(intervals.values[index])) {
            return false;
        }
    }
    for (uint32_t index = 1U; index < intervals.count; ++index) {
        const G1RootReachInterval value = intervals.values[index];
        uint32_t destination = index;
        while (destination > 0U) {
            const G1RootReachInterval& previous =
                intervals.values[destination - 1U];
            const bool before_previous =
                value.lower_m < previous.lower_m ||
                (value.lower_m == previous.lower_m &&
                 value.upper_m < previous.upper_m);
            if (!before_previous) break;
            intervals.values[destination] = previous;
            --destination;
        }
        intervals.values[destination] = value;
    }
    return true;
}

static bool g1_root_reach_interval_set_append(
    G1RootReachIntervalSet& intervals,
    double lower_m,
    double upper_m)
{
    const G1RootReachInterval value = {lower_m, upper_m};
    if (!g1_root_reach_interval_is_valid(value) ||
        intervals.count >= 4U) {
        return false;
    }
    intervals.values[intervals.count] = value;
    ++intervals.count;
    return true;
}

static bool g1_root_reach_intersect_interval_sets(
    G1RootReachIntervalSet& output,
    const G1RootReachIntervalSet& left,
    const G1RootReachIntervalSet& right)
{
    if (left.count > 4U || right.count > 4U) return false;
    G1RootReachIntervalSet candidate = {};
    for (uint32_t left_index = 0U;
         left_index < left.count;
         ++left_index) {
        const G1RootReachInterval& left_value =
            left.values[left_index];
        if (!g1_root_reach_interval_is_valid(left_value)) return false;
        for (uint32_t right_index = 0U;
             right_index < right.count;
             ++right_index) {
            const G1RootReachInterval& right_value =
                right.values[right_index];
            if (!g1_root_reach_interval_is_valid(right_value)) return false;
            const double lower =
                left_value.lower_m > right_value.lower_m
                    ? left_value.lower_m
                    : right_value.lower_m;
            const double upper =
                left_value.upper_m < right_value.upper_m
                    ? left_value.upper_m
                    : right_value.upper_m;
            if (lower <= upper &&
                !g1_root_reach_interval_set_append(
                    candidate, lower, upper)) {
                return false;
            }
        }
    }
    if (!g1_root_reach_interval_set_sort(candidate)) return false;
    output = candidate;
    return true;
}

static bool g1_root_reach_foot_intervals(
    G1RootReachIntervalSet& output,
    const G1RootReachFootGeometry& geometry)
{
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    double x_squared = 0.0;
    double z_squared = 0.0;
    double horizontal_squared = 0.0;
    double minimum_squared = 0.0;
    double maximum_squared = 0.0;
    if (!geometry.active ||
        !g1_root_reach_checked_subtract(
            static_cast<double>(geometry.baseline_ankle_target.x),
            static_cast<double>(geometry.hip.x), x) ||
        !g1_root_reach_checked_subtract(
            static_cast<double>(geometry.baseline_ankle_target.y),
            static_cast<double>(geometry.hip.y), y) ||
        !g1_root_reach_checked_subtract(
            static_cast<double>(geometry.baseline_ankle_target.z),
            static_cast<double>(geometry.hip.z), z) ||
        !g1_root_reach_checked_multiply(x, x, x_squared) ||
        !g1_root_reach_checked_multiply(z, z, z_squared) ||
        !g1_root_reach_checked_add(
            x_squared, z_squared, horizontal_squared) ||
        !g1_root_reach_checked_multiply(
            geometry.shell.minimum_distance_m,
            geometry.shell.minimum_distance_m,
            minimum_squared) ||
        !g1_root_reach_checked_multiply(
            geometry.shell.maximum_distance_m,
            geometry.shell.maximum_distance_m,
            maximum_squared)) {
        return false;
    }

    G1RootReachIntervalSet candidate = {};
    if (horizontal_squared > maximum_squared) {
        output = candidate;
        return true;
    }
    double maximum_vertical_squared = 0.0;
    if (!g1_root_reach_checked_subtract(
            maximum_squared,
            horizontal_squared,
            maximum_vertical_squared) ||
        maximum_vertical_squared < 0.0) {
        return false;
    }
    const volatile double maximum_vertical =
        std::sqrt(maximum_vertical_squared);
    double maximum_lower = 0.0;
    double maximum_upper = 0.0;
    if (!terrain_double_is_finite(maximum_vertical) ||
        !g1_root_reach_checked_subtract(
            y, maximum_vertical, maximum_lower) ||
        !g1_root_reach_checked_add(
            y, maximum_vertical, maximum_upper)) {
        return false;
    }

    if (minimum_squared <= horizontal_squared) {
        if (!g1_root_reach_interval_set_append(
                candidate, maximum_lower, maximum_upper)) {
            return false;
        }
    } else {
        double minimum_vertical_squared = 0.0;
        if (!g1_root_reach_checked_subtract(
                minimum_squared,
                horizontal_squared,
                minimum_vertical_squared) ||
            minimum_vertical_squared <= 0.0) {
            return false;
        }
        const volatile double minimum_vertical =
            std::sqrt(minimum_vertical_squared);
        double hole_lower = 0.0;
        double hole_upper = 0.0;
        if (!terrain_double_is_finite(minimum_vertical) ||
            !g1_root_reach_checked_subtract(
                y, minimum_vertical, hole_lower) ||
            !g1_root_reach_checked_add(
                y, minimum_vertical, hole_upper)) {
            return false;
        }
        const double left_upper =
            hole_lower < maximum_upper ? hole_lower : maximum_upper;
        if (maximum_lower <= left_upper &&
            !g1_root_reach_interval_set_append(
                candidate, maximum_lower, left_upper)) {
            return false;
        }
        const double right_lower =
            hole_upper > maximum_lower ? hole_upper : maximum_lower;
        if (right_lower <= maximum_upper &&
            !g1_root_reach_interval_set_append(
                candidate, right_lower, maximum_upper)) {
            return false;
        }
    }
    if (!g1_root_reach_interval_set_sort(candidate)) return false;
    output = candidate;
    return true;
}

static G1RootReachCandidateStatus g1_root_reach_candidate_revalidate(
    const G1RootReachFootGeometry (&feet)[2],
    const slice1d<vec3> baseline_positions,
    const slice1d<quat> baseline_rotations,
    const slice1d<int> parents,
    float delta_m)
{
    vec3 local_positions[G1_BoneCount];
    vec3 global_positions[G1_BoneCount];
    quat global_rotations[G1_BoneCount];
    for (int bone = 0; bone < G1_BoneCount; ++bone) {
        local_positions[bone] = baseline_positions.data[bone];
    }
    const bool applied =
        (terrain_float_bits(delta_m) & UINT32_C(0x7fffffff)) != 0U;
    const G1RootReachPlan plan = {
        true, true, applied, applied ? delta_m : 0.0f
    };
    float adjusted_root_y = 0.0f;
    if (!g1_apply_root_reach_plan_y(
            adjusted_root_y,
            baseline_positions.data[G1_Simulation].y,
            plan)) {
        return G1RootReachCandidateInvalid;
    }
    local_positions[G1_Simulation].y = adjusted_root_y;
    if (!g1_root_reach_checked_forward_kinematics(
            global_positions,
            global_rotations,
            slice1d<vec3>(G1_BoneCount, local_positions),
            baseline_rotations,
            parents)) {
        return G1RootReachCandidateInvalid;
    }
    for (int foot = 0; foot < 2; ++foot) {
        if (!feet[foot].active) continue;
        const G1LegConfig config = foot == 0
            ? g1_left_leg_config()
            : g1_right_leg_config();
        G1PhysicalSolePositionTarget physical_target = {};
        IKTargetProjection projection = {};
        if (!g1_physical_sole_position_target(
                physical_target,
                global_positions[config.contact],
                global_rotations[config.contact],
                global_positions[config.ankle],
                config,
                feet[foot].desired_sole_center,
                feet[foot].desired_sole_normal,
                NULL,
                0) ||
            !ik_project_target(
                projection,
                global_positions[config.hip],
                global_positions[config.knee],
                global_positions[config.ankle],
                physical_target.ankle_target,
                config.reach_buffer_m)) {
            return G1RootReachCandidateInvalid;
        }
        if (!projection.reachable ||
            !g1_ik_vec3_bits_equal(
                projection.clamped_target,
                physical_target.ankle_target)) {
            return G1RootReachCandidateRejected;
        }
    }
    return G1RootReachCandidateAccepted;
}

static bool g1_root_reach_materialize_interval_candidate(
    float& output,
    const G1RootReachInterval& interval)
{
    if (!g1_root_reach_interval_is_valid(interval)) return false;
    if (interval.lower_m <= 0.0 && interval.upper_m >= 0.0) {
        output = 0.0f;
        return true;
    }
    const bool negative = interval.upper_m < 0.0;
    const double endpoint =
        negative ? interval.upper_m : interval.lower_m;
    float candidate = 0.0f;
    if (!ik_checked_binary32_commit(endpoint, candidate)) return false;
    double promoted = static_cast<double>(candidate);
    if (promoted < interval.lower_m) {
        candidate = std::nextafter(
            candidate, std::numeric_limits<float>::infinity());
        promoted = static_cast<double>(candidate);
    } else if (promoted > interval.upper_m) {
        candidate = std::nextafter(
            candidate, -std::numeric_limits<float>::infinity());
        promoted = static_cast<double>(candidate);
    }
    if (!g1_ik_float_is_runtime_value(candidate) ||
        promoted < interval.lower_m || promoted > interval.upper_m ||
        std::fabs(promoted) >
            static_cast<double>(G1RootReachMaximumAdjustmentM)) {
        return false;
    }
    if ((terrain_float_bits(candidate) & UINT32_C(0x7fffffff)) == 0U) {
        candidate = 0.0f;
    }
    output = candidate;
    return true;
}

static bool g1_root_reach_cursor_precedes(
    const G1RootReachCandidateCursor& left,
    const G1RootReachCandidateCursor& right)
{
    const double left_distance =
        std::fabs(static_cast<double>(left.delta_m));
    const double right_distance =
        std::fabs(static_cast<double>(right.delta_m));
    if (left_distance != right_distance) {
        return left_distance < right_distance;
    }
    if (left.interval.lower_m != right.interval.lower_m) {
        return left.interval.lower_m < right.interval.lower_m;
    }
    if (left.interval.upper_m != right.interval.upper_m) {
        return left.interval.upper_m < right.interval.upper_m;
    }
    return left.interval_index < right.interval_index;
}

#if defined(G1_IK_ENABLE_TEST_SEAMS)

static bool g1_root_reach_audit_record_cursor(
    G1RootReachPlannerAudit* audit,
    uint32_t interval_index,
    float initial_delta_m)
{
    if (audit == NULL) return true;
    const uint32_t capacity = static_cast<uint32_t>(
        sizeof(audit->cursors) / sizeof(audit->cursors[0]));
    if (audit->cursor_count >= capacity) return false;
    G1RootReachAuditCursor& cursor =
        audit->cursors[audit->cursor_count];
    cursor.interval_index = interval_index;
    cursor.initial_delta_bits = terrain_float_bits(initial_delta_m);
    ++audit->cursor_count;
    return true;
}

static bool g1_root_reach_audit_record_attempt(
    G1RootReachPlannerAudit* audit,
    uint32_t cursor_index,
    float delta_m,
    G1RootReachCandidateStatus status)
{
    if (audit == NULL) return true;
    const uint32_t capacity = static_cast<uint32_t>(
        sizeof(audit->attempts) / sizeof(audit->attempts[0]));
    if (audit->attempt_count >= capacity ||
        (status != G1RootReachCandidateRejected &&
         status != G1RootReachCandidateAccepted)) {
        return false;
    }
    G1RootReachAuditAttempt& attempt =
        audit->attempts[audit->attempt_count];
    attempt.cursor_index = cursor_index;
    attempt.delta_bits = terrain_float_bits(delta_m);
    attempt.status = status == G1RootReachCandidateAccepted
        ? G1RootReachAuditAccepted
        : G1RootReachAuditRejected;
    ++audit->attempt_count;
    return true;
}

#endif

static bool g1_root_reach_ranges_overlap_inputs(
    const void* owner,
    size_t owner_bytes,
    const slice1d<vec3> positions,
    const slice1d<quat> rotations,
    const slice1d<int> parents,
    const slice1d<bool> contacts,
    const G1FootTarget& left_target,
    const G1FootTarget& right_target)
{
    return g1_ik_memory_ranges_overlap(
               owner, owner_bytes,
               positions.data,
               positions.data == NULL
                   ? 0U
                   : static_cast<size_t>(G1_BoneCount) * sizeof(vec3)) ||
           g1_ik_memory_ranges_overlap(
               owner, owner_bytes,
               rotations.data,
               rotations.data == NULL
                   ? 0U
                   : static_cast<size_t>(G1_BoneCount) * sizeof(quat)) ||
           g1_ik_memory_ranges_overlap(
               owner, owner_bytes,
               parents.data,
               parents.data == NULL
                   ? 0U
                   : static_cast<size_t>(G1_BoneCount) * sizeof(int)) ||
           g1_ik_memory_ranges_overlap(
               owner, owner_bytes,
               contacts.data,
               contacts.data == NULL ? 0U : 2U * sizeof(bool)) ||
           g1_ik_memory_ranges_overlap(
               owner, owner_bytes,
               &left_target, sizeof(left_target)) ||
           g1_ik_memory_ranges_overlap(
               owner, owner_bytes,
               &right_target, sizeof(right_target));
}

} // namespace

bool g1_root_reach_plan_is_valid(const G1RootReachPlan& plan)
{
    if (!g1_ik_float_is_runtime_value(plan.root_y_delta_m) ||
        std::fabs(static_cast<double>(plan.root_y_delta_m)) >
            static_cast<double>(G1RootReachMaximumAdjustmentM)) {
        return false;
    }
    const uint32_t delta_bits = terrain_float_bits(plan.root_y_delta_m);
    const bool positive_zero = delta_bits == 0U;
    if (!plan.active) {
        return !plan.common_interval_found && !plan.applied && positive_zero;
    }
    if (!plan.common_interval_found) {
        return !plan.applied && positive_zero;
    }
    if (!plan.applied) return positive_zero;
    return (delta_bits & UINT32_C(0x7fffffff)) != 0U;
}

namespace
{

static bool g1_plan_recorded_contact_root_reach_implementation(
    G1RootReachPlan& output,
    const slice1d<vec3> baseline_positions,
    const slice1d<quat> baseline_rotations,
    const slice1d<int> parents,
    const slice1d<bool> recorded_contacts,
    const G1FootTarget& left_target,
    const G1FootTarget& right_target,
    char* error,
    int error_capacity
#if defined(G1_IK_ENABLE_TEST_SEAMS)
    , G1RootReachPlannerAudit* audit
#endif
    )
{
    const size_t error_bytes =
        error != NULL && error_capacity > 0
            ? static_cast<size_t>(error_capacity)
            : 0U;
    if (error_capacity < 0 ||
        g1_root_reach_ranges_overlap_inputs(
            &output, sizeof(output),
            baseline_positions, baseline_rotations,
            parents, recorded_contacts,
            left_target, right_target) ||
        g1_ik_memory_ranges_overlap(
            error, error_bytes, &output, sizeof(output)) ||
        g1_root_reach_ranges_overlap_inputs(
            error, error_bytes,
            baseline_positions, baseline_rotations,
            parents, recorded_contacts,
            left_target, right_target)) {
        return false;
    }
    if (baseline_positions.size != G1_BoneCount ||
        baseline_positions.data == NULL ||
        baseline_rotations.size != G1_BoneCount ||
        baseline_rotations.data == NULL ||
        parents.size != G1_BoneCount || parents.data == NULL ||
        recorded_contacts.size != 2 || recorded_contacts.data == NULL ||
        !g1_ik_parent_topology_validate(parents, error, error_capacity) ||
        !g1_foot_target_is_valid(left_target) ||
        !g1_foot_target_is_valid(right_target)) {
        return g1_ik_error(
            error, error_capacity,
            "G1 root reach planner requires a valid complete frame");
    }

    const bool active =
        recorded_contacts.data[0] || recorded_contacts.data[1];
    if (!active) {
        const G1RootReachPlan inactive = {};
        output = inactive;
        return true;
    }

    vec3 global_positions[G1_BoneCount];
    quat global_rotations[G1_BoneCount];
    if (!g1_root_reach_checked_forward_kinematics(
            global_positions,
            global_rotations,
            baseline_positions,
            baseline_rotations,
            parents)) {
        return g1_ik_error(
            error, error_capacity,
            "G1 root reach baseline FK failed");
    }

    const G1LegConfig configs[2] = {
        g1_left_leg_config(), g1_right_leg_config()
    };
    const G1FootTarget* const targets[2] = {
        &left_target, &right_target
    };
    G1RootReachFootGeometry feet[2] = {};
    G1RootReachIntervalSet common = {};
    common.count = 1U;
    common.values[0].lower_m =
        -static_cast<double>(G1RootReachMaximumAdjustmentM);
    common.values[0].upper_m =
        static_cast<double>(G1RootReachMaximumAdjustmentM);

    for (int foot = 0; foot < 2; ++foot) {
        feet[foot].active = recorded_contacts.data[foot];
        if (!feet[foot].active) continue;
        const G1LegConfig& config = configs[foot];
        const G1FootTarget& target = *targets[foot];
        if (!g1_ik_surface_normal_is_valid(
                target.desired_sole_normal)) {
            return g1_ik_error(
                error, error_capacity,
                "G1 root reach recorded contact target is malformed");
        }
        G1PhysicalSolePositionTarget physical_target = {};
        if (!g1_physical_sole_position_target(
                physical_target,
                global_positions[config.contact],
                global_rotations[config.contact],
                global_positions[config.ankle],
                config,
                target.sole_center,
                target.desired_sole_normal,
                error,
                error_capacity) ||
            !ik_effective_reach_shell(
                feet[foot].shell,
                global_positions[config.hip],
                global_positions[config.knee],
                global_positions[config.ankle],
                config.reach_buffer_m)) {
            return g1_ik_error(
                error, error_capacity,
                "G1 root reach foot geometry failed");
        }
        feet[foot].hip = global_positions[config.hip];
        feet[foot].knee = global_positions[config.knee];
        feet[foot].ankle = global_positions[config.ankle];
        feet[foot].baseline_ankle_target =
            physical_target.ankle_target;
        feet[foot].desired_sole_center = target.sole_center;
        feet[foot].desired_sole_normal =
            target.desired_sole_normal;

        G1RootReachIntervalSet foot_intervals = {};
        G1RootReachIntervalSet intersection = {};
        if (!g1_root_reach_foot_intervals(
                foot_intervals, feet[foot]) ||
            !g1_root_reach_intersect_interval_sets(
                intersection, common, foot_intervals)) {
            return g1_ik_error(
                error, error_capacity,
                "G1 root reach interval arithmetic failed");
        }
        common = intersection;
    }

    const G1RootReachPlan unavailable = {
        true, false, false, 0.0f
    };
    if (common.count == 0U) {
        output = unavailable;
        return true;
    }

    G1RootReachCandidateCursor cursors[4] = {};
    for (uint32_t index = 0U; index < common.count; ++index) {
        cursors[index].interval_index = index;
        cursors[index].interval = common.values[index];
        if (g1_root_reach_materialize_interval_candidate(
                cursors[index].delta_m,
                cursors[index].interval)) {
            cursors[index].active = true;
#if defined(G1_IK_ENABLE_TEST_SEAMS)
            if (!g1_root_reach_audit_record_cursor(
                    audit, index, cursors[index].delta_m)) {
                return g1_ik_error(
                    error, error_capacity,
                    "G1 root reach audit cursor capacity exceeded");
            }
#endif
        }
    }

    for (uint32_t candidate_test = 0U;
         candidate_test < G1RootReachMaximumBoundaryCandidateTests;
         ++candidate_test) {
        uint32_t selected = 4U;
        for (uint32_t index = 0U; index < common.count; ++index) {
            if (!cursors[index].active) continue;
            if (selected == 4U ||
                g1_root_reach_cursor_precedes(
                    cursors[index], cursors[selected])) {
                selected = index;
            }
        }
        if (selected == 4U) break;

        G1RootReachCandidateCursor& cursor = cursors[selected];
        const G1RootReachCandidateStatus status =
            g1_root_reach_candidate_revalidate(
                feet,
                baseline_positions,
                baseline_rotations,
                parents,
                cursor.delta_m);
        if (status == G1RootReachCandidateInvalid) {
            return g1_ik_error(
                error, error_capacity,
                "G1 root reach production revalidation failed");
        }
#if defined(G1_IK_ENABLE_TEST_SEAMS)
        if (!g1_root_reach_audit_record_attempt(
                audit,
                cursor.interval_index,
                cursor.delta_m,
                status)) {
            return g1_ik_error(
                error, error_capacity,
                "G1 root reach audit attempt capacity exceeded");
        }
#endif
        if (status == G1RootReachCandidateAccepted) {
            const bool applied =
                (terrain_float_bits(cursor.delta_m) &
                 UINT32_C(0x7fffffff)) != 0U;
            const G1RootReachPlan candidate = {
                true, true, applied,
                applied ? cursor.delta_m : 0.0f
            };
            if (!g1_root_reach_plan_is_valid(candidate)) {
                return g1_ik_error(
                    error, error_capacity,
                    "G1 root reach plan validation failed");
            }
            output = candidate;
            return true;
        }

        if (cursor.interval.lower_m <= 0.0 &&
            cursor.interval.upper_m >= 0.0) {
            cursor.active = false;
            continue;
        }
        const float direction = cursor.interval.upper_m < 0.0
            ? -std::numeric_limits<float>::infinity()
            : std::numeric_limits<float>::infinity();
        const float inward = std::nextafter(cursor.delta_m, direction);
        const double promoted = static_cast<double>(inward);
        if (!g1_ik_float_is_runtime_value(inward) ||
            promoted < cursor.interval.lower_m ||
            promoted > cursor.interval.upper_m ||
            std::fabs(promoted) > static_cast<double>(
                G1RootReachMaximumAdjustmentM)) {
            cursor.active = false;
        } else {
            cursor.delta_m = inward;
        }
    }

    output = unavailable;
    return true;
}

} // namespace

bool g1_plan_recorded_contact_root_reach(
    G1RootReachPlan& output,
    const slice1d<vec3> baseline_positions,
    const slice1d<quat> baseline_rotations,
    const slice1d<int> parents,
    const slice1d<bool> recorded_contacts,
    const G1FootTarget& left_target,
    const G1FootTarget& right_target,
    char* error,
    int error_capacity)
{
    return g1_plan_recorded_contact_root_reach_implementation(
        output,
        baseline_positions,
        baseline_rotations,
        parents,
        recorded_contacts,
        left_target,
        right_target,
        error,
        error_capacity
#if defined(G1_IK_ENABLE_TEST_SEAMS)
        , NULL
#endif
        );
}

#if defined(G1_IK_ENABLE_TEST_SEAMS)

bool g1_plan_recorded_contact_root_reach_audited(
    G1RootReachPlan& output,
    G1RootReachPlannerAudit& audit,
    const slice1d<vec3> baseline_positions,
    const slice1d<quat> baseline_rotations,
    const slice1d<int> parents,
    const slice1d<bool> recorded_contacts,
    const G1FootTarget& left_target,
    const G1FootTarget& right_target,
    char* error,
    int error_capacity)
{
    const size_t error_bytes =
        error != NULL && error_capacity > 0
            ? static_cast<size_t>(error_capacity)
            : 0U;
    if (error_capacity < 0 ||
        g1_ik_memory_ranges_overlap(
            &audit, sizeof(audit), &output, sizeof(output)) ||
        g1_root_reach_ranges_overlap_inputs(
            &audit, sizeof(audit),
            baseline_positions, baseline_rotations,
            parents, recorded_contacts,
            left_target, right_target) ||
        g1_ik_memory_ranges_overlap(
            error, error_bytes, &audit, sizeof(audit))) {
        return false;
    }

    G1RootReachPlannerAudit candidate = {};
    if (!g1_plan_recorded_contact_root_reach_implementation(
            output,
            baseline_positions,
            baseline_rotations,
            parents,
            recorded_contacts,
            left_target,
            right_target,
            error,
            error_capacity,
            &candidate)) {
        return false;
    }
    audit = candidate;
    return true;
}

#endif

bool g1_apply_root_reach_plan_y(
    float& output_root_y,
    float baseline_root_y,
    const G1RootReachPlan& plan)
{
    if (g1_ik_memory_ranges_overlap(
            &output_root_y, sizeof(output_root_y),
            &plan, sizeof(plan)) ||
        !g1_root_reach_plan_is_valid(plan) ||
        !g1_ik_float_is_runtime_value(baseline_root_y)) {
        return false;
    }
    if (!plan.applied) {
        output_root_y = baseline_root_y;
        return true;
    }
    double sum = 0.0;
    float rounded = 0.0f;
    if (!g1_root_reach_checked_add(
            static_cast<double>(baseline_root_y),
            static_cast<double>(plan.root_y_delta_m),
            sum) ||
        !ik_checked_binary32_commit(sum, rounded) ||
        terrain_float_bits(rounded) ==
            terrain_float_bits(baseline_root_y)) {
        return false;
    }
    output_root_y = rounded;
    return true;
}
