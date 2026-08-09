#pragma once

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-result"
#endif
#include "g1_command_runtime.h"
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif
#include "g1_ik.h"

#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

enum G1FootprintStatus
{
    G1FootprintOk,
    G1FootprintOutsideDomain,
    G1FootprintBudgetExceeded,
    G1FootprintInvalidInput,
    G1FootprintInvalidField,
    G1FootprintArithmeticFailure,
};

struct G1FootprintBudget
{
    uint32_t maximum_sweeps = 24;
    uint32_t maximum_surface_queries = 35;
    uint32_t maximum_node_visits = 65536;
};

struct G1FootprintWork
{
    uint32_t sweeps = 0;
    uint32_t surface_queries = 0;
    uint32_t node_visits = 0;
};

struct G1FootContactSchedule
{
    bool contact[2][G1CommandTrajectorySampleCount] = {};
};

struct G1FootprintProbe
{
    vec3 current_sphere_center;
    vec3 current_sole_point;
    G1SurfaceSample current_surface;
    vec3 predicted_sphere_centers[G1CommandTrajectorySampleCount];
    vec3 predicted_sole_points[G1CommandTrajectorySampleCount];
    G1SurfaceQueryStatus predicted_surface_status[
        G1CommandTrajectorySampleCount] = {
            G1SurfaceQueryInvalid, G1SurfaceQueryInvalid,
            G1SurfaceQueryInvalid, G1SurfaceQueryInvalid};
    G1SurfaceSample predicted_surfaces[G1CommandTrajectorySampleCount];
    G1SurfaceSample selected_landing_surface;
    float corridor_minimum_height = 0.0f;
    float corridor_maximum_height = 0.0f;
    int encountered_walkability_class = 1;
};

struct G1FootprintFootObservation
{
    G1FootprintProbe probes[4];
    bool current_contact = false;
    bool landing_expected = false;
    bool landing_patch_ready = false;
    uint32_t landing_sample = UINT32_MAX;
    vec3 predicted_landing_sole_center;
    G1SurfaceQueryStatus predicted_landing_surface_status =
        G1SurfaceQueryInvalid;
    G1SurfaceSample predicted_landing_surface;
    int predicted_landing_walkability_class = 0;
    double landing_patch_maximum_residual_m = 0.0;
    float corridor_minimum_height = 0.0f;
    float corridor_maximum_height = 0.0f;
    double maximum_root_split_m = 0.0;
    int encountered_walkability_class = 1;
    bool multilevel = false;
};

struct G1FootprintObservation
{
    G1SurfaceSample root_surface;
    G1FootprintFootObservation feet[2];
    bool blocked = false;
    walkability_reason blocked_reason = walkability_clear;
    G1FootprintWork work;
};

static inline G1FootprintBudget g1_footprint_budget()
{
    return G1FootprintBudget{};
}

static inline bool g1_footprint_checked_byte_count(
    std::size_t& output,
    int count,
    std::size_t element_size)
{
    output = 0;
    if (count < 0 || element_size == 0) return false;
    if (static_cast<std::size_t>(count) >
        std::numeric_limits<std::size_t>::max() / element_size) {
        return false;
    }
    output = static_cast<std::size_t>(count) * element_size;
    return true;
}

static inline bool g1_footprint_checked_2d_byte_count(
    std::size_t& output,
    int rows,
    int columns,
    std::size_t element_size)
{
    output = 0;
    if (rows < 0 || columns < 0) return false;
    std::size_t elements = 0;
    if (!terrain_size_multiply(
            static_cast<std::size_t>(rows),
            static_cast<std::size_t>(columns),
            elements) ||
        elements > std::numeric_limits<std::size_t>::max() / element_size) {
        return false;
    }
    output = elements * element_size;
    return true;
}

static inline bool g1_footprint_overlaps(
    const void* first,
    std::size_t first_size,
    const void* second,
    std::size_t second_size)
{
    return g1_command_memory_overlaps(
        first, first_size, second, second_size);
}

static inline bool g1_footprint_diagnostic_aliases(
    const void* memory,
    std::size_t size,
    char* error,
    int error_capacity)
{
    return error != NULL && error_capacity > 0 &&
           g1_footprint_overlaps(
               memory,
               size,
               error,
               static_cast<std::size_t>(error_capacity));
}

static inline bool g1_footprint_schedule_failure(
    const G1FootContactSchedule& output,
    char* error,
    int error_capacity,
    const char* message)
{
    if (!g1_footprint_diagnostic_aliases(
            &output, sizeof(output), error, error_capacity)) {
        terrain_error(error, error_capacity, "%s", message);
    }
    return false;
}

static inline G1FootprintStatus g1_footprint_observation_failure(
    const G1FootprintObservation& output,
    G1FootprintStatus status,
    char* error,
    int error_capacity,
    const char* message)
{
    if (!g1_footprint_diagnostic_aliases(
            &output, sizeof(output), error, error_capacity)) {
        terrain_error(error, error_capacity, "%s", message);
    }
    return status;
}

static inline bool g1_foot_contact_schedule_build(
    G1FootContactSchedule& output,
    const slice2d<bool> database_contacts,
    const slice1d<int> range_starts,
    const slice1d<int> range_stops,
    int current_frame,
    bool current_left_contact,
    bool current_right_contact,
    float dt,
    float trajectory_sample_time,
    char* error,
    int error_capacity)
{
    std::size_t contact_bytes = 0;
    std::size_t start_bytes = 0;
    std::size_t stop_bytes = 0;
    const bool contact_size_valid = g1_footprint_checked_2d_byte_count(
        contact_bytes,
        database_contacts.rows,
        database_contacts.cols,
        sizeof(bool));
    const bool start_size_valid = g1_footprint_checked_byte_count(
        start_bytes, range_starts.size, sizeof(int));
    const bool stop_size_valid = g1_footprint_checked_byte_count(
        stop_bytes, range_stops.size, sizeof(int));
    if (error_capacity < 0 ||
        g1_footprint_diagnostic_aliases(
            &output, sizeof(output), error, error_capacity) ||
        (contact_size_valid && database_contacts.data != NULL &&
         g1_footprint_overlaps(
             &output, sizeof(output),
             database_contacts.data, contact_bytes)) ||
        (start_size_valid && range_starts.data != NULL &&
         g1_footprint_overlaps(
             &output, sizeof(output), range_starts.data, start_bytes)) ||
        (stop_size_valid && range_stops.data != NULL &&
         g1_footprint_overlaps(
             &output, sizeof(output), range_stops.data, stop_bytes))) {
        return false;
    }
    if (!contact_size_valid || !start_size_valid || !stop_size_valid ||
        database_contacts.rows <= 0 || database_contacts.cols != 2 ||
        database_contacts.data == NULL ||
        range_starts.size <= 0 ||
        range_starts.size != range_stops.size ||
        range_starts.data == NULL || range_stops.data == NULL ||
        current_frame < 0 ||
        !g1_dt_is_exact_60_hz(dt) ||
        !terrain_float_is_positive_normal(trajectory_sample_time)) {
        return g1_footprint_schedule_failure(
            output,
            error,
            error_capacity,
            "invalid G1 footprint contact schedule input");
    }

    int containing_range = -1;
    int expected_start = 0;
    for (int range = 0; range < range_starts.size; ++range) {
        const int start = range_starts(range);
        const int stop = range_stops(range);
        if (start != expected_start || stop <= start ||
            stop > database_contacts.rows) {
            return g1_footprint_schedule_failure(
                output,
                error,
                error_capacity,
                "G1 footprint contact ranges must be contiguous half-open rows");
        }
        if (current_frame >= start && current_frame < stop) {
            containing_range = range;
        }
        expected_start = stop;
    }
    if (expected_start != database_contacts.rows || containing_range < 0) {
        return g1_footprint_schedule_failure(
            output,
            error,
            error_capacity,
            "G1 footprint current frame is outside the complete range table");
    }

    G1FootContactSchedule candidate;
    const int range_stop = range_stops(containing_range);
    const int final_frame = range_stop - 1;
    for (int sample = 0; sample < G1CommandTrajectorySampleCount; ++sample) {
        int offset = 0;
        if (sample > 0) {
            float horizon = 0.0f;
            float ratio = 0.0f;
            if (!terrain_f32_mul(
                    horizon,
                    static_cast<float>(sample),
                    trajectory_sample_time) ||
                !terrain_f32_div(ratio, horizon, dt)) {
                return g1_footprint_schedule_failure(
                    output,
                    error,
                    error_capacity,
                    "G1 footprint contact horizon arithmetic failed");
            }
            const float rounded = ceilf(ratio);
            if (!terrain_float_is_finite(rounded) || rounded < 1.0f ||
                static_cast<double>(rounded) >
                    static_cast<double>(INT_MAX)) {
                return g1_footprint_schedule_failure(
                    output,
                    error,
                    error_capacity,
                    "G1 footprint contact horizon exceeds integer range");
            }
            offset = static_cast<int>(rounded);
        }
        const int remaining = final_frame - current_frame;
        const int frame = offset > remaining
            ? final_frame
            : current_frame + offset;
        candidate.contact[0][sample] = database_contacts(frame, 0);
        candidate.contact[1][sample] = database_contacts(frame, 1);
    }
    if (candidate.contact[0][0] != current_left_contact ||
        candidate.contact[1][0] != current_right_contact) {
        return g1_footprint_schedule_failure(
            output,
            error,
            error_capacity,
            "G1 footprint current contact bits disagree with schedule sample zero");
    }
    output = candidate;
    return true;
}

static inline int g1_footprint_class_merge(int current, int incoming)
{
    if (current == 0 || incoming == 0) return 0;
    return current > incoming ? current : incoming;
}

static inline G1FootprintStatus g1_footprint_surface_query(
    G1SurfaceSample& output,
    G1SurfaceQueryStatus& query_status,
    G1FootprintWork& work,
    uint32_t surface_limit,
    const heightfield& field,
    float x,
    float z)
{
    if (work.surface_queries >= surface_limit) {
        return G1FootprintBudgetExceeded;
    }
    ++work.surface_queries;
    G1SurfaceSample candidate = {};
    const G1SurfaceQueryStatus status =
        g1_surface_query_v2(candidate, field, x, z);
    query_status = status;
    if (status == G1SurfaceQueryOutside) {
        return G1FootprintOutsideDomain;
    }
    if (status != G1SurfaceQueryValid) {
        return G1FootprintInvalidField;
    }
    output = candidate;
    return G1FootprintOk;
}

static inline G1FootprintStatus g1_footprint_nearest_walkability_axis(
    int& output,
    float input,
    float origin,
    float cell_size,
    int count)
{
    float canonical = 0.0f;
    if (count < 2 || !terrain_float_is_normal_or_positive_zero(origin) ||
        !terrain_float_is_positive_normal(cell_size) ||
        !terrain_v2_query_coordinate(input, canonical)) {
        return G1FootprintArithmeticFailure;
    }
    const volatile double maximum_product =
        static_cast<double>(count - 1) * static_cast<double>(cell_size);
    const volatile double maximum =
        static_cast<double>(origin) + maximum_product;
    if (!terrain_double_is_finite(maximum)) {
        return G1FootprintArithmeticFailure;
    }
    const double value = static_cast<double>(canonical);
    if (value < static_cast<double>(origin) || value > maximum) {
        return G1FootprintOutsideDomain;
    }
    const volatile double difference =
        value - static_cast<double>(origin);
    const volatile double normalized =
        difference / static_cast<double>(cell_size);
    const volatile double shifted = normalized + 0.5;
    const double rounded = std::floor(shifted);
    if (!terrain_double_is_finite(normalized) ||
        !terrain_double_is_finite(shifted) ||
        !terrain_double_is_finite(rounded) || rounded < 0.0 ||
        rounded > static_cast<double>(INT_MAX)) {
        return G1FootprintArithmeticFailure;
    }
    int candidate = static_cast<int>(rounded);
    if (candidate >= count) candidate = count - 1;
    output = candidate;
    return G1FootprintOk;
}

static inline G1FootprintStatus g1_footprint_walkability_class_at(
    int& output,
    const walkability_grid& grid,
    const heightfield& field,
    float input_x,
    float input_z)
{
    if (!walkability_grid_matches_heightfield(grid, field)) {
        return G1FootprintInvalidField;
    }
    int node_x = 0;
    int node_z = 0;
    G1FootprintStatus status = g1_footprint_nearest_walkability_axis(
        node_x, input_x, field.origin_x, field.cell_size, grid.nx);
    if (status != G1FootprintOk) return status;
    status = g1_footprint_nearest_walkability_axis(
        node_z, input_z, field.origin_z, field.cell_size, grid.nz);
    if (status != G1FootprintOk) return status;
    int candidate = 0;
    if (!walkability_cell_value(candidate, grid, node_x, node_z)) {
        return G1FootprintInvalidField;
    }
    output = candidate;
    return G1FootprintOk;
}

struct G1FootprintNodeWindow
{
    int x0 = 0;
    int x1 = 0;
    int z0 = 0;
    int z1 = 0;
    uint32_t count = 0;
};

static inline bool g1_footprint_segment_node_window(
    G1FootprintNodeWindow& output,
    const heightfield& field,
    vec3 start,
    vec3 stop)
{
    const double radius = static_cast<double>(0.02f);
    const double start_x = static_cast<double>(start.x);
    const double stop_x = static_cast<double>(stop.x);
    const double start_z = static_cast<double>(start.z);
    const double stop_z = static_cast<double>(stop.z);
    const double minimum_x =
        (start_x < stop_x ? start_x : stop_x) - radius;
    const double maximum_x =
        (start_x > stop_x ? start_x : stop_x) + radius;
    const double minimum_z =
        (start_z < stop_z ? start_z : stop_z) - radius;
    const double maximum_z =
        (start_z > stop_z ? start_z : stop_z) + radius;
    const double inverse_cell = 1.0 / static_cast<double>(field.cell_size);
    double minimum_grid_x =
        (minimum_x - static_cast<double>(field.origin_x)) * inverse_cell;
    double maximum_grid_x =
        (maximum_x - static_cast<double>(field.origin_x)) * inverse_cell;
    double minimum_grid_z =
        (minimum_z - static_cast<double>(field.origin_z)) * inverse_cell;
    double maximum_grid_z =
        (maximum_z - static_cast<double>(field.origin_z)) * inverse_cell;
    if (!terrain_double_is_finite(minimum_grid_x) ||
        !terrain_double_is_finite(maximum_grid_x) ||
        !terrain_double_is_finite(minimum_grid_z) ||
        !terrain_double_is_finite(maximum_grid_z)) {
        return false;
    }
    const double last_x = static_cast<double>(field.nx - 1);
    const double last_z = static_cast<double>(field.nz - 1);
    minimum_grid_x = minimum_grid_x < 0.0 ? 0.0 :
        (minimum_grid_x > last_x ? last_x : minimum_grid_x);
    maximum_grid_x = maximum_grid_x < 0.0 ? 0.0 :
        (maximum_grid_x > last_x ? last_x : maximum_grid_x);
    minimum_grid_z = minimum_grid_z < 0.0 ? 0.0 :
        (minimum_grid_z > last_z ? last_z : minimum_grid_z);
    maximum_grid_z = maximum_grid_z < 0.0 ? 0.0 :
        (maximum_grid_z > last_z ? last_z : maximum_grid_z);

    G1FootprintNodeWindow candidate;
    if (!walkability_checked_floor_to_int(
            candidate.x0, minimum_grid_x, 0, field.nx - 1) ||
        !walkability_checked_ceil_to_int(
            candidate.x1, maximum_grid_x, 0, field.nx - 1) ||
        !walkability_checked_floor_to_int(
            candidate.z0, minimum_grid_z, 0, field.nz - 1) ||
        !walkability_checked_ceil_to_int(
            candidate.z1, maximum_grid_z, 0, field.nz - 1)) {
        return false;
    }
    if (candidate.x0 > 0) --candidate.x0;
    if (candidate.z0 > 0) --candidate.z0;
    if (candidate.x1 < field.nx - 1) ++candidate.x1;
    if (candidate.z1 < field.nz - 1) ++candidate.z1;
    const std::size_t width =
        static_cast<std::size_t>(candidate.x1 - candidate.x0) + 1U;
    const std::size_t depth =
        static_cast<std::size_t>(candidate.z1 - candidate.z0) + 1U;
    std::size_t count = 0;
    if (!terrain_size_multiply(width, depth, count) ||
        count > static_cast<std::size_t>(UINT32_MAX)) {
        return false;
    }
    candidate.count = static_cast<uint32_t>(count);
    output = candidate;
    return true;
}

static inline bool g1_footprint_transport_point(
    vec3 output[G1CommandTrajectorySampleCount],
    vec3 current,
    const G1CommandSnapshot& command)
{
    output[0] = current;
    const vec3 difference =
        current - command.predicted_root_positions[0];
    if (!g1_ik_vec3_is_runtime_value(difference)) return false;
    const vec3 root_local = quat_inv_mul_vec3(
        command.predicted_root_rotations[0], difference);
    if (!g1_ik_vec3_is_runtime_value(root_local)) return false;
    for (int sample = 1; sample < G1CommandTrajectorySampleCount; ++sample) {
        const vec3 rotated = quat_mul_vec3(
            command.predicted_root_rotations[sample], root_local);
        if (!g1_ik_vec3_is_runtime_value(rotated)) return false;
        output[sample] =
            command.predicted_root_positions[sample] + rotated;
        if (!g1_ik_vec3_is_runtime_value(output[sample])) return false;
    }
    return true;
}

static inline void g1_footprint_apply_contact_phase(
    vec3 output[G1CommandTrajectorySampleCount],
    const vec3 transported[G1CommandTrajectorySampleCount],
    const bool contact[G1CommandTrajectorySampleCount])
{
    output[0] = transported[0];
    vec3 anchor = transported[0];
    for (int sample = 1; sample < G1CommandTrajectorySampleCount; ++sample) {
        if (contact[sample]) {
            if (!contact[sample - 1]) anchor = transported[sample];
            output[sample] = anchor;
        } else {
            output[sample] = transported[sample];
        }
    }
}

static inline bool g1_footprint_observation_aliases_inputs(
    const G1FootprintObservation& output,
    const G1FootprintBudget& limits,
    const heightfield& field,
    const walkability_grid& grid,
    const G1CommandSnapshot& command,
    const G1FootContactSchedule& contacts,
    const slice1d<vec3> global_positions,
    const slice1d<quat> global_rotations,
    char* error,
    int error_capacity)
{
    const void* const output_memory = &output;
    const std::size_t output_size = sizeof(output);
    if (g1_footprint_diagnostic_aliases(
            output_memory, output_size, error, error_capacity) ||
        g1_footprint_overlaps(
            output_memory, output_size, &limits, sizeof(limits)) ||
        g1_footprint_overlaps(
            output_memory, output_size, &field, sizeof(field)) ||
        g1_footprint_overlaps(
            output_memory, output_size, &grid, sizeof(grid)) ||
        g1_footprint_overlaps(
            output_memory, output_size, &command, sizeof(command)) ||
        g1_footprint_overlaps(
            output_memory, output_size, &contacts, sizeof(contacts))) {
        return true;
    }
    std::size_t position_bytes = 0;
    std::size_t rotation_bytes = 0;
    std::size_t height_bytes = 0;
    std::size_t cell_bytes = 0;
    const bool position_size_valid = g1_footprint_checked_byte_count(
        position_bytes, global_positions.size, sizeof(vec3));
    const bool rotation_size_valid = g1_footprint_checked_byte_count(
        rotation_bytes, global_rotations.size, sizeof(quat));
    const bool height_size_valid = g1_footprint_checked_byte_count(
        height_bytes, field.heights.size, sizeof(float));
    const bool cell_size_valid = g1_footprint_checked_byte_count(
        cell_bytes, grid.cells.size, sizeof(uint8_t));
    return
        (position_size_valid && global_positions.data != NULL &&
         g1_footprint_overlaps(
             output_memory, output_size,
             global_positions.data, position_bytes)) ||
        (rotation_size_valid && global_rotations.data != NULL &&
         g1_footprint_overlaps(
             output_memory, output_size,
             global_rotations.data, rotation_bytes)) ||
        (height_size_valid && field.heights.data != NULL &&
         g1_footprint_overlaps(
             output_memory, output_size,
             field.heights.data, height_bytes)) ||
        (cell_size_valid && grid.cells.data != NULL &&
         g1_footprint_overlaps(
             output_memory, output_size,
             grid.cells.data, cell_bytes));
}

static inline G1FootprintStatus g1_footprint_observe_v2(
    G1FootprintObservation& output,
    const G1FootprintBudget& limits,
    const heightfield& field,
    const walkability_grid& grid,
    const G1CommandSnapshot& command,
    const G1FootContactSchedule& contacts,
    const slice1d<vec3> global_positions,
    const slice1d<quat> global_rotations,
    char* error,
    int error_capacity)
{
    static_assert(G1CommandTrajectorySampleCount == 4,
                  "G1 footprint observation requires four samples");
    static_assert(sizeof(float) == sizeof(uint32_t),
                  "G1 footprint thresholds require binary32");
    if (error_capacity < 0 ||
        g1_footprint_observation_aliases_inputs(
            output,
            limits,
            field,
            grid,
            command,
            contacts,
            global_positions,
            global_rotations,
            error,
            error_capacity)) {
        return G1FootprintInvalidInput;
    }
    if (field.version != 2 || !terrain_heightfield_is_queryable(field) ||
        !walkability_grid_matches_heightfield(grid, field)) {
        return g1_footprint_observation_failure(
            output,
            G1FootprintInvalidField,
            error,
            error_capacity,
            "invalid G1HF/v2 footprint field or walkability grid");
    }
    if (!g1_command_snapshot_is_valid(command) ||
        global_positions.size != G1_BoneCount ||
        global_rotations.size != G1_BoneCount ||
        global_positions.data == NULL || global_rotations.data == NULL) {
        return g1_footprint_observation_failure(
            output,
            G1FootprintInvalidInput,
            error,
            error_capacity,
            "invalid G1 footprint command or exact 31-bone pose shape");
    }
    for (int bone = 0; bone < G1_BoneCount; ++bone) {
        if (!g1_ik_vec3_is_runtime_value(global_positions(bone)) ||
            !ik_quat_is_unit(global_rotations(bone))) {
            return g1_footprint_observation_failure(
                output,
                G1FootprintInvalidInput,
                error,
                error_capacity,
                "invalid G1 footprint global pose value");
        }
    }

    G1FootprintObservation candidate;
    const G1LegConfig legs[2] = {
        g1_left_leg_config(), g1_right_leg_config()
    };
    for (int foot_index = 0; foot_index < 2; ++foot_index) {
        G1FootprintFootObservation& foot = candidate.feet[foot_index];
        foot.current_contact = contacts.contact[foot_index][0];
        for (int probe_index = 0; probe_index < 4; ++probe_index) {
            G1FootprintProbe& probe = foot.probes[probe_index];
            const quat ankle_rotation =
                global_rotations(legs[foot_index].ankle);
            const vec3 sphere_offset = quat_mul_vec3(
                ankle_rotation,
                legs[foot_index].foot_sphere_centers_local[probe_index]);
            const vec3 sole_offset = quat_mul_vec3(
                ankle_rotation,
                legs[foot_index].sole_points_local[probe_index]);
            if (!g1_ik_vec3_is_runtime_value(sphere_offset) ||
                !g1_ik_vec3_is_runtime_value(sole_offset)) {
                return g1_footprint_observation_failure(
                    output,
                    G1FootprintArithmeticFailure,
                    error,
                    error_capacity,
                    "G1 footprint ankle rotation arithmetic failed");
            }
            probe.current_sphere_center =
                global_positions(legs[foot_index].ankle) + sphere_offset;
            probe.current_sole_point =
                global_positions(legs[foot_index].ankle) + sole_offset;
            if (!g1_ik_vec3_is_runtime_value(probe.current_sphere_center) ||
                !g1_ik_vec3_is_runtime_value(probe.current_sole_point)) {
                return g1_footprint_observation_failure(
                    output,
                    G1FootprintArithmeticFailure,
                    error,
                    error_capacity,
                    "G1 footprint current geometry arithmetic failed");
            }
            vec3 transported_spheres[G1CommandTrajectorySampleCount];
            vec3 transported_soles[G1CommandTrajectorySampleCount];
            if (!g1_footprint_transport_point(
                    transported_spheres,
                    probe.current_sphere_center,
                    command) ||
                !g1_footprint_transport_point(
                    transported_soles,
                    probe.current_sole_point,
                    command)) {
                return g1_footprint_observation_failure(
                    output,
                    G1FootprintArithmeticFailure,
                    error,
                    error_capacity,
                    "G1 footprint root transport arithmetic failed");
            }
            g1_footprint_apply_contact_phase(
                probe.predicted_sphere_centers,
                transported_spheres,
                contacts.contact[foot_index]);
            g1_footprint_apply_contact_phase(
                probe.predicted_sole_points,
                transported_soles,
                contacts.contact[foot_index]);
        }
    }

    uint32_t landing_count = 0;
    for (int foot_index = 0; foot_index < 2; ++foot_index) {
        G1FootprintFootObservation& foot = candidate.feet[foot_index];
        if (foot.current_contact) continue;
        for (int sample = 1; sample < G1CommandTrajectorySampleCount; ++sample) {
            if (!contacts.contact[foot_index][sample - 1] &&
                contacts.contact[foot_index][sample]) {
                foot.landing_expected = true;
                foot.landing_sample = static_cast<uint32_t>(sample);
                ++landing_count;
                break;
            }
        }
    }
    const uint32_t required_surface_queries = 33U + landing_count;
    const uint32_t sweep_limit =
        limits.maximum_sweeps < 24U ? limits.maximum_sweeps : 24U;
    const uint32_t surface_limit =
        limits.maximum_surface_queries < 35U
            ? limits.maximum_surface_queries
            : 35U;
    const uint32_t node_limit =
        limits.maximum_node_visits < 65536U
            ? limits.maximum_node_visits
            : 65536U;
    if (sweep_limit < 24U || surface_limit < required_surface_queries) {
        return g1_footprint_observation_failure(
            output,
            G1FootprintBudgetExceeded,
            error,
            error_capacity,
            "G1 footprint fixed sweep or surface budget exceeded");
    }

    G1SurfaceQueryStatus root_status = G1SurfaceQueryInvalid;
    G1FootprintStatus status = g1_footprint_surface_query(
        candidate.root_surface,
        root_status,
        candidate.work,
        surface_limit,
        field,
        command.predicted_root_positions[0].x,
        command.predicted_root_positions[0].z);
    if (status != G1FootprintOk) {
        return g1_footprint_observation_failure(
            output, status, error, error_capacity,
            "G1 footprint root surface query failed");
    }

    for (int foot_index = 0; foot_index < 2; ++foot_index) {
        for (int probe_index = 0; probe_index < 4; ++probe_index) {
            G1FootprintProbe& probe =
                candidate.feet[foot_index].probes[probe_index];
            G1SurfaceQueryStatus current_status = G1SurfaceQueryInvalid;
            status = g1_footprint_surface_query(
                probe.current_surface,
                current_status,
                candidate.work,
                surface_limit,
                field,
                probe.current_sole_point.x,
                probe.current_sole_point.z);
            if (status != G1FootprintOk) {
                return g1_footprint_observation_failure(
                    output, status, error, error_capacity,
                    "G1 footprint current sole surface query failed");
            }
            probe.predicted_surface_status[0] = current_status;
            probe.predicted_surfaces[0] = probe.current_surface;
            probe.corridor_minimum_height = probe.current_surface.height;
            probe.corridor_maximum_height = probe.current_surface.height;
            for (int sample = 1;
                 sample < G1CommandTrajectorySampleCount;
                 ++sample) {
                status = g1_footprint_surface_query(
                    probe.predicted_surfaces[sample],
                    probe.predicted_surface_status[sample],
                    candidate.work,
                    surface_limit,
                    field,
                    probe.predicted_sole_points[sample].x,
                    probe.predicted_sole_points[sample].z);
                if (status != G1FootprintOk) {
                    return g1_footprint_observation_failure(
                        output, status, error, error_capacity,
                        "G1 footprint future sole surface query failed");
                }
            }
        }
    }

    for (int foot_index = 0; foot_index < 2; ++foot_index) {
        G1FootprintFootObservation& foot = candidate.feet[foot_index];
        for (int probe_index = 0; probe_index < 4; ++probe_index) {
            G1FootprintProbe& probe = foot.probes[probe_index];
            for (int segment = 0;
                 segment + 1 < G1CommandTrajectorySampleCount;
                 ++segment) {
                if (candidate.work.sweeps >= sweep_limit) {
                    return g1_footprint_observation_failure(
                        output,
                        G1FootprintBudgetExceeded,
                        error,
                        error_capacity,
                        "G1 footprint sweep budget exceeded");
                }
                ++candidate.work.sweeps;
                const vec3 start = probe.predicted_sole_points[segment];
                const vec3 stop = probe.predicted_sole_points[segment + 1];
                const walkability_sweep_result sweep =
                    walkability_sweep(grid, field, start, stop, 0.02f);
                probe.encountered_walkability_class =
                    g1_footprint_class_merge(
                        probe.encountered_walkability_class,
                        sweep.encountered_class);
                if (sweep.blocked) {
                    if (sweep.reason == walkability_out_of_bounds) {
                        return g1_footprint_observation_failure(
                            output,
                            G1FootprintOutsideDomain,
                            error,
                            error_capacity,
                            "G1 footprint sweep left terrain domain");
                    }
                    if (sweep.reason == walkability_nonfinite) {
                        return g1_footprint_observation_failure(
                            output,
                            G1FootprintInvalidField,
                            error,
                            error_capacity,
                            "G1 footprint sweep encountered malformed field data");
                    }
                    if (sweep.reason == walkability_blocked_cell &&
                        !candidate.blocked) {
                        candidate.blocked = true;
                        candidate.blocked_reason = sweep.reason;
                    }
                }

                G1FootprintNodeWindow window;
                if (!g1_footprint_segment_node_window(
                        window, field, start, stop)) {
                    return g1_footprint_observation_failure(
                        output,
                        G1FootprintArithmeticFailure,
                        error,
                        error_capacity,
                        "G1 footprint node-window arithmetic failed");
                }
                if (window.count > node_limit - candidate.work.node_visits) {
                    return g1_footprint_observation_failure(
                        output,
                        G1FootprintBudgetExceeded,
                        error,
                        error_capacity,
                        "G1 footprint aggregate node budget exceeded");
                }
                for (int z = window.z0;;) {
                    for (int x = window.x0;;) {
                        const int node = z * field.nx + x;
                        const float height = field.heights(node);
                        ++candidate.work.node_visits;
                        if (!terrain_float_is_normal_or_positive_zero(height)) {
                            return g1_footprint_observation_failure(
                                output,
                                G1FootprintInvalidField,
                                error,
                                error_capacity,
                                "G1 footprint visited malformed height node");
                        }
                        int cell_class = 0;
                        if (!walkability_cell_value(
                                cell_class, grid, x, z)) {
                            return g1_footprint_observation_failure(
                                output,
                                G1FootprintInvalidField,
                                error,
                                error_capacity,
                                "G1 footprint visited malformed walkability node");
                        }
                        if (height < probe.corridor_minimum_height) {
                            probe.corridor_minimum_height = height;
                        }
                        if (height > probe.corridor_maximum_height) {
                            probe.corridor_maximum_height = height;
                        }
                        if (x == window.x1) break;
                        ++x;
                    }
                    if (z == window.z1) break;
                    ++z;
                }
            }
        }
    }

    for (int foot_index = 0; foot_index < 2; ++foot_index) {
        G1FootprintFootObservation& foot = candidate.feet[foot_index];
        foot.corridor_minimum_height =
            foot.probes[0].corridor_minimum_height;
        foot.corridor_maximum_height =
            foot.probes[0].corridor_maximum_height;
        foot.encountered_walkability_class =
            foot.probes[0].encountered_walkability_class;
        for (int probe_index = 0; probe_index < 4; ++probe_index) {
            const G1FootprintProbe& probe = foot.probes[probe_index];
            if (probe.corridor_minimum_height <
                foot.corridor_minimum_height) {
                foot.corridor_minimum_height =
                    probe.corridor_minimum_height;
            }
            if (probe.corridor_maximum_height >
                foot.corridor_maximum_height) {
                foot.corridor_maximum_height =
                    probe.corridor_maximum_height;
            }
            foot.encountered_walkability_class =
                g1_footprint_class_merge(
                    foot.encountered_walkability_class,
                    probe.encountered_walkability_class);
            const double differences[] = {
                std::fabs(
                    static_cast<double>(candidate.root_surface.height) -
                    static_cast<double>(probe.current_surface.height)),
                std::fabs(
                    static_cast<double>(candidate.root_surface.height) -
                    static_cast<double>(probe.corridor_minimum_height)),
                std::fabs(
                    static_cast<double>(candidate.root_surface.height) -
                    static_cast<double>(probe.corridor_maximum_height)),
            };
            for (const double difference : differences) {
                if (!terrain_double_is_finite(difference)) {
                    return g1_footprint_observation_failure(
                        output,
                        G1FootprintArithmeticFailure,
                        error,
                        error_capacity,
                        "G1 footprint root split arithmetic failed");
                }
                if (difference > foot.maximum_root_split_m) {
                    foot.maximum_root_split_m = difference;
                }
            }
        }
        foot.multilevel = foot.maximum_root_split_m >=
            static_cast<double>(0.04f);
    }

    for (int foot_index = 0; foot_index < 2; ++foot_index) {
        G1FootprintFootObservation& foot = candidate.feet[foot_index];
        if (!foot.landing_expected) continue;
        const uint32_t sample = foot.landing_sample;
        volatile double sum_x = 0.0;
        volatile double sum_y = 0.0;
        volatile double sum_z = 0.0;
        for (int probe_index = 0; probe_index < 4; ++probe_index) {
            const vec3 point =
                foot.probes[probe_index].predicted_sole_points[sample];
            sum_x = sum_x + static_cast<double>(point.x);
            sum_y = sum_y + static_cast<double>(point.y);
            sum_z = sum_z + static_cast<double>(point.z);
            foot.probes[probe_index].selected_landing_surface =
                foot.probes[probe_index].predicted_surfaces[sample];
        }
        const volatile double centroid_x = sum_x / 4.0;
        const volatile double centroid_y = sum_y / 4.0;
        const volatile double centroid_z = sum_z / 4.0;
        if (!terrain_v2_round_output(
                centroid_x, foot.predicted_landing_sole_center.x) ||
            !terrain_v2_round_output(
                centroid_y, foot.predicted_landing_sole_center.y) ||
            !terrain_v2_round_output(
                centroid_z, foot.predicted_landing_sole_center.z)) {
            return g1_footprint_observation_failure(
                output,
                G1FootprintArithmeticFailure,
                error,
                error_capacity,
                "G1 footprint landing centroid arithmetic failed");
        }
        status = g1_footprint_surface_query(
            foot.predicted_landing_surface,
            foot.predicted_landing_surface_status,
            candidate.work,
            surface_limit,
            field,
            foot.predicted_landing_sole_center.x,
            foot.predicted_landing_sole_center.z);
        if (status != G1FootprintOk) {
            return g1_footprint_observation_failure(
                output, status, error, error_capacity,
                "G1 footprint landing-center surface query failed");
        }
        status = g1_footprint_walkability_class_at(
            foot.predicted_landing_walkability_class,
            grid,
            field,
            foot.predicted_landing_sole_center.x,
            foot.predicted_landing_sole_center.z);
        if (status != G1FootprintOk) {
            return g1_footprint_observation_failure(
                output, status, error, error_capacity,
                "G1 footprint landing centroid walkability lookup failed");
        }
        if (foot.predicted_landing_walkability_class == 0 &&
            !candidate.blocked) {
            candidate.blocked = true;
            candidate.blocked_reason = walkability_blocked_cell;
        }

        const vec3 center_point(
            foot.predicted_landing_sole_center.x,
            foot.predicted_landing_surface.height,
            foot.predicted_landing_sole_center.z);
        double maximum_residual = 0.0;
        bool all_class_one =
            foot.predicted_landing_walkability_class == 1;
        for (int probe_index = 0; probe_index < 4; ++probe_index) {
            const G1FootprintProbe& probe = foot.probes[probe_index];
            all_class_one = all_class_one &&
                probe.encountered_walkability_class == 1 &&
                probe.predicted_surface_status[sample] ==
                    G1SurfaceQueryValid;
            const vec3 point(
                probe.predicted_sole_points[sample].x,
                probe.predicted_surfaces[sample].height,
                probe.predicted_sole_points[sample].z);
            const volatile double difference_x =
                static_cast<double>(point.x) -
                static_cast<double>(center_point.x);
            const volatile double difference_y =
                static_cast<double>(point.y) -
                static_cast<double>(center_point.y);
            const volatile double difference_z =
                static_cast<double>(point.z) -
                static_cast<double>(center_point.z);
            const volatile double x_term = difference_x *
                static_cast<double>(foot.predicted_landing_surface.normal.x);
            const volatile double y_term = difference_y *
                static_cast<double>(foot.predicted_landing_surface.normal.y);
            const volatile double z_term = difference_z *
                static_cast<double>(foot.predicted_landing_surface.normal.z);
            const volatile double first_sum = x_term + y_term;
            const volatile double residual = std::fabs(first_sum + z_term);
            if (!terrain_double_is_finite(residual)) {
                return g1_footprint_observation_failure(
                    output,
                    G1FootprintArithmeticFailure,
                    error,
                    error_capacity,
                    "G1 footprint landing residual arithmetic failed");
            }
            if (residual > maximum_residual) {
                maximum_residual = residual;
            }
        }
        foot.landing_patch_maximum_residual_m = maximum_residual;
        const float exact_residual_limit = []() {
            const uint32_t threshold_bits = UINT32_C(0x3ba3d70a);
            float threshold = 0.0f;
            std::memcpy(&threshold, &threshold_bits, sizeof(threshold));
            return threshold;
        }();
        foot.landing_patch_ready = all_class_one &&
            maximum_residual <= static_cast<double>(exact_residual_limit);
    }

    output = candidate;
    return G1FootprintOk;
}
