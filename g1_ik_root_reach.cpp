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
