#pragma once

#include "common.h"
#include "vec.h"
#include "quat.h"
#include "array.h"

#include <assert.h>
#include <float.h>
#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <utility>

//--------------------------------------

enum
{
    BOUND_SM_SIZE = 16,
    BOUND_LR_SIZE = 64,
};

struct database
{
    array2d<vec3> bone_positions;
    array2d<vec3> bone_velocities;
    array2d<quat> bone_rotations;
    array2d<vec3> bone_angular_velocities;
    array1d<int> bone_parents;
    
    array1d<int> range_starts;
    array1d<int> range_stops;
    
    array2d<float> features;
    array1d<float> features_offset;
    array1d<float> features_scale;
    array2d<float> terrain_features;
    
    array2d<bool> contact_states;
    
    array2d<float> bound_sm_min;
    array2d<float> bound_sm_max;
    array2d<float> bound_lr_min;
    array2d<float> bound_lr_max;
    
    int nframes() const { return bone_positions.rows; }
    int nbones() const { return bone_positions.cols; }
    int nranges() const { return range_starts.size; }
    int nfeatures() const { return features.cols; }
    int ncontacts() const { return contact_states.cols; }
};

static inline uint32_t feature_float_bits(float value);
static inline bool feature_float_is_finite(float value);
static inline bool feature_float_is_positive_finite(float value);
void database_build_bounds(database& db);

void database_load(database& db, const char* filename)
{
    FILE* f = fopen(filename, "rb");
    assert(f != NULL);
    
    array2d_read(db.bone_positions, f);
    array2d_read(db.bone_velocities, f);
    array2d_read(db.bone_rotations, f);
    array2d_read(db.bone_angular_velocities, f);
    array1d_read(db.bone_parents, f);
    
    array1d_read(db.range_starts, f);
    array1d_read(db.range_stops, f);
    
    array2d_read(db.contact_states, f);
    
    fclose(f);
}

void database_save_matching_features(const database& db, const char* filename)
{
    FILE* f = fopen(filename, "wb");
    assert(f != NULL);
    
    array2d_write(db.features, f);
    array1d_write(db.features_offset, f);
    array1d_write(db.features_scale, f);
    
    fclose(f);
}

static inline bool database_read_exact(
    FILE* file, void* output, const size_t bytes)
{
    return bytes == 0 || fread(output, 1, bytes, file) == bytes;
}

static inline bool database_read_u32_le(FILE* file, uint32_t& output)
{
    unsigned char bytes[4] = {};
    if (!database_read_exact(file, bytes, sizeof(bytes))) return false;
    output = static_cast<uint32_t>(bytes[0]) |
        (static_cast<uint32_t>(bytes[1]) << 8) |
        (static_cast<uint32_t>(bytes[2]) << 16) |
        (static_cast<uint32_t>(bytes[3]) << 24);
    return true;
}

static inline bool database_load_matching_features_checked(
    database& db,
    const char* filename,
    char* error,
    const int capacity)
{
    FILE* file = filename == NULL ? NULL : fopen(filename, "rb");
    if (file == NULL) {
        if (error != NULL && capacity > 0) {
            snprintf(error, static_cast<size_t>(capacity),
                     "cannot open matching features");
        }
        return false;
    }
    uint32_t rows = 0, columns = 0, offsets = 0, scales = 0;
    const bool header_ok = database_read_u32_le(file, rows) &&
        database_read_u32_le(file, columns) &&
        rows == static_cast<uint32_t>(db.nframes()) && columns == 31 &&
        rows <= static_cast<uint32_t>(INT32_MAX);
    array2d<float> values;
    array1d<float> loaded_offsets;
    array1d<float> loaded_scales;
    bool ok = header_ok;
    if (ok) {
        values.resize(static_cast<int>(rows), static_cast<int>(columns));
        ok = database_read_exact(
            file,
            values.data,
            static_cast<size_t>(rows) * columns * sizeof(float)) &&
            database_read_u32_le(file, offsets) && offsets == columns;
    }
    if (ok) {
        loaded_offsets.resize(static_cast<int>(offsets));
        ok = database_read_exact(
            file, loaded_offsets.data,
            static_cast<size_t>(offsets) * sizeof(float)) &&
            database_read_u32_le(file, scales) && scales == columns;
    }
    if (ok) {
        loaded_scales.resize(static_cast<int>(scales));
        ok = database_read_exact(
            file, loaded_scales.data,
            static_cast<size_t>(scales) * sizeof(float)) &&
            fgetc(file) == EOF && !ferror(file);
    }
    fclose(file);
    for (int row = 0; ok && row < values.rows; ++row) {
        for (int column = 0; column < values.cols; ++column) {
            ok = feature_float_is_finite(values(row, column));
            if (!ok) break;
        }
    }
    for (int column = 0; ok && column < loaded_offsets.size; ++column) {
        ok = feature_float_is_finite(loaded_offsets(column)) &&
            (feature_float_is_positive_finite(loaded_scales(column)) ||
             feature_float_bits(loaded_scales(column)) ==
                 feature_float_bits(FLT_MAX));
    }
    if (!ok) {
        if (error != NULL && capacity > 0) {
            snprintf(error, static_cast<size_t>(capacity),
                     "matching features are malformed or incompatible");
        }
        return false;
    }
    db.features = std::move(values);
    db.features_offset = std::move(loaded_offsets);
    db.features_scale = std::move(loaded_scales);
    database_build_bounds(db);
    return true;
}

// When we add an offset to a frame in the database there is a chance
// it will go out of the relevant range so here we can clamp it to 
// the last frame of that range.
int database_trajectory_index_clamp(database& db, int frame, int offset)
{
    for (int i = 0; i < db.nranges(); i++)
    {
        if (frame >= db.range_starts(i) && frame < db.range_stops(i))
        {
            return clamp(frame + offset, db.range_starts(i), db.range_stops(i) - 1);
        }
    }
    
    assert(false);
    return -1;
}

//--------------------------------------

static inline uint32_t feature_float_bits(const float value)
{
    uint32_t bits = 0;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static inline bool database_rotation_continuity_validate(
    const database& db,
    const float maximum_step,
    char* error,
    const int capacity,
    float* observed_maximum = NULL)
{
    if (!feature_float_is_positive_finite(maximum_step) ||
        db.bone_rotations.rows != db.nframes() ||
        db.bone_rotations.cols != db.nbones() || db.nbones() <= 1 ||
        db.range_starts.size <= 0 ||
        db.range_starts.size != db.range_stops.size)
    {
        if (error != NULL && capacity > 0) {
            snprintf(error, static_cast<size_t>(capacity),
                     "database rotation continuity inputs are invalid");
        }
        return false;
    }
    float candidate_maximum = 0.0f;
    for (int range = 0; range < db.nranges(); ++range) {
        const int start = db.range_starts(range);
        const int stop = db.range_stops(range);
        if (start < 0 || stop <= start || stop > db.nframes()) {
            if (error != NULL && capacity > 0) {
                snprintf(error, static_cast<size_t>(capacity),
                         "database range %d is invalid", range);
            }
            return false;
        }
        for (int frame = start + 1; frame < stop; ++frame) {
            for (int bone = 1; bone < db.nbones(); ++bone) {
                const float step = quat_angle_between(
                    db.bone_rotations(frame - 1, bone),
                    db.bone_rotations(frame, bone));
                if (!feature_float_is_finite(step) ||
                    step > maximum_step)
                {
                    if (error != NULL && capacity > 0) {
                        snprintf(
                            error,
                            static_cast<size_t>(capacity),
                            "database rotation discontinuity range=%d "
                            "frame=%d bone=%d step=%.9g limit=%.9g",
                            range, frame, bone, step, maximum_step);
                    }
                    return false;
                }
                candidate_maximum = maxf(candidate_maximum, step);
            }
        }
    }
    if (observed_maximum != NULL) *observed_maximum = candidate_maximum;
    return true;
}

static inline bool feature_float_is_finite(const float value)
{
    return (feature_float_bits(value) & UINT32_C(0x7f800000)) !=
           UINT32_C(0x7f800000);
}

static inline bool feature_float_is_positive_finite(const float value)
{
    const uint32_t bits = feature_float_bits(value);
    return (bits & UINT32_C(0x80000000)) == 0 &&
           (bits & UINT32_C(0x7fffffff)) != 0 &&
           (bits & UINT32_C(0x7f800000)) != UINT32_C(0x7f800000);
}

static inline bool feature_weight_is_valid(const float weight)
{
    const uint32_t bits = feature_float_bits(weight);
    return (bits & UINT32_C(0x7f800000)) != UINT32_C(0x7f800000) &&
           ((bits & UINT32_C(0x80000000)) == 0 ||
            (bits & UINT32_C(0x7fffffff)) == 0);
}

static inline void disable_feature_group(
    slice2d<float> features,
    slice1d<float> features_offset,
    slice1d<float> features_scale,
    const int offset,
    const int size)
{
    for (int j = 0; j < size; ++j)
    {
        if (!feature_float_is_finite(features_offset(offset + j)))
        {
            features_offset(offset + j) = 0.0f;
        }
        features_scale(offset + j) = FLT_MAX;
    }

    for (int i = 0; i < features.rows; ++i)
    {
        for (int j = 0; j < size; ++j)
        {
            features(i, offset + j) = 0.0f;
        }
    }
}

void normalize_feature(
    slice2d<float> features,
    slice1d<float> features_offset,
    slice1d<float> features_scale,
    const int offset, 
    const int size, 
    const float weight = 1.0f)
{
    assert(feature_weight_is_valid(weight));
    if (!feature_weight_is_valid(weight))
    {
        return;
    }

    bool has_variation = false;
    for (int j = 0; j < size && !has_variation; ++j)
    {
        const uint32_t first = feature_float_bits(features(0, offset + j));
        for (int i = 1; i < features.rows; ++i)
        {
            if (feature_float_bits(features(i, offset + j)) != first)
            {
                has_variation = true;
                break;
            }
        }
    }

    // First compute what is essentially the mean 
    // value for each feature dimension
    for (int j = 0; j < size; j++)
    {
        features_offset(offset + j) = 0.0f;    
    }
    
    for (int i = 0; i < features.rows; i++)
    {
        for (int j = 0; j < size; j++)
        {
            features_offset(offset + j) += features(i, offset + j) / features.rows;
        }
    }
    
    // A zero-weight group is explicitly disabled. Keep its raw mean for safe
    // denormalization, but do not evaluate variance or any FLT_MAX arithmetic.
    if (weight == 0.0f)
    {
        disable_feature_group(
            features, features_offset, features_scale, offset, size);
        return;
    }

    assert(has_variation);
    if (!has_variation)
    {
        disable_feature_group(
            features, features_offset, features_scale, offset, size);
        return;
    }

    // Now compute the variance of each feature dimension
    array1d<float> vars(size);
    vars.zero();
    
    for (int i = 0; i < features.rows; i++)
    {
        for (int j = 0; j < size; j++)
        {
            vars(j) += squaref(features(i, offset + j) - features_offset(offset + j)) / features.rows;
        }
    }
    
    // We compute the overall std of the feature as the average
    // std across all dimensions
    float std = 0.0f;
    for (int j = 0; j < size; j++)
    {
        std += sqrtf(vars(j)) / size;
    }
    
    // Features with no variation can have zero std which is
    // almost always a bug.
    assert(feature_float_is_positive_finite(std));
    if (!feature_float_is_positive_finite(std))
    {
        disable_feature_group(
            features, features_offset, features_scale, offset, size);
        return;
    }

    const float scale = std / weight;
    const bool scale_valid =
        feature_float_is_positive_finite(scale) &&
        feature_float_bits(scale) != feature_float_bits(FLT_MAX);
    assert(scale_valid);
    if (!scale_valid)
    {
        disable_feature_group(
            features, features_offset, features_scale, offset, size);
        return;
    }
    
    // The scale of a feature is just the std divided by the weight
    for (int j = 0; j < size; j++)
    {
        features_scale(offset + j) = scale;
    }
    
    // Using the offset and scale we can then normalize the features
    for (int i = 0; i < features.rows; i++)
    {
        for (int j = 0; j < size; j++)
        {
            features(i, offset + j) = (features(i, offset + j) - features_offset(offset + j)) / features_scale(offset + j);
        }
    }
}

static inline bool feature_scale_is_disabled(const float scale)
{
    return feature_float_bits(scale) == feature_float_bits(FLT_MAX);
}

static inline float normalize_query_feature(
    const float value,
    const float offset,
    const float scale)
{
    return feature_scale_is_disabled(scale)
        ? 0.0f
        : (value - offset) / scale;
}

void denormalize_features(
    slice1d<float> features,
    const slice1d<float> features_offset,
    const slice1d<float> features_scale)
{
    for (int i = 0; i < features.size; i++)
    {
        features(i) = feature_scale_is_disabled(features_scale(i))
            ? features_offset(i)
            : (features(i) * features_scale(i)) + features_offset(i);
    }  
}

//--------------------------------------

// Here I am using a simple recursive version of forward kinematics
void forward_kinematics(
    vec3& bone_position,
    quat& bone_rotation,
    const slice1d<vec3> bone_positions,
    const slice1d<quat> bone_rotations,
    const slice1d<int> bone_parents,
    const int bone)
{
    if (bone_parents(bone) != -1)
    {
        vec3 parent_position;
        quat parent_rotation;
        
        forward_kinematics(
            parent_position,
            parent_rotation,
            bone_positions,
            bone_rotations,
            bone_parents,
            bone_parents(bone));
        
        bone_position = quat_mul_vec3(parent_rotation, bone_positions(bone)) + parent_position;
        bone_rotation = quat_mul(parent_rotation, bone_rotations(bone));
    }
    else
    {
        bone_position = bone_positions(bone);
        bone_rotation = bone_rotations(bone); 
    }
}

// Forward kinematics but also compute the velocities
void forward_kinematics_velocity(
    vec3& bone_position,
    vec3& bone_velocity,
    quat& bone_rotation,
    vec3& bone_angular_velocity,
    const slice1d<vec3> bone_positions,
    const slice1d<vec3> bone_velocities,
    const slice1d<quat> bone_rotations,
    const slice1d<vec3> bone_angular_velocities,
    const slice1d<int> bone_parents,
    const int bone)
{
    //
    if (bone_parents(bone) != -1)
    {
        vec3 parent_position;
        vec3 parent_velocity;
        quat parent_rotation;
        vec3 parent_angular_velocity;
        
        forward_kinematics_velocity(
            parent_position,
            parent_velocity,
            parent_rotation,
            parent_angular_velocity,
            bone_positions,
            bone_velocities,
            bone_rotations,
            bone_angular_velocities,
            bone_parents,
            bone_parents(bone));
        
        bone_position = quat_mul_vec3(parent_rotation, bone_positions(bone)) + parent_position;
        bone_velocity = 
            parent_velocity + 
            quat_mul_vec3(parent_rotation, bone_velocities(bone)) + 
            cross(parent_angular_velocity, quat_mul_vec3(parent_rotation, bone_positions(bone)));
        bone_rotation = quat_mul(parent_rotation, bone_rotations(bone));
        bone_angular_velocity = quat_mul_vec3(parent_rotation, bone_angular_velocities(bone)) + parent_angular_velocity;
    }
    else
    {
        bone_position = bone_positions(bone);
        bone_velocity = bone_velocities(bone);
        bone_rotation = bone_rotations(bone);
        bone_angular_velocity = bone_angular_velocities(bone); 
    }
}

// Compute forward kinematics for all joints
void forward_kinematics_full(
    slice1d<vec3> global_bone_positions,
    slice1d<quat> global_bone_rotations,
    const slice1d<vec3> local_bone_positions,
    const slice1d<quat> local_bone_rotations,
    const slice1d<int> bone_parents)
{
    for (int i = 0; i < bone_parents.size; i++)
    {
        // Assumes bones are always sorted from root onwards
        assert(bone_parents(i) < i);
        
        if (bone_parents(i) == -1)
        {
            global_bone_positions(i) = local_bone_positions(i);
            global_bone_rotations(i) = local_bone_rotations(i);
        }
        else
        {
            vec3 parent_position = global_bone_positions(bone_parents(i));
            quat parent_rotation = global_bone_rotations(bone_parents(i));
            global_bone_positions(i) = quat_mul_vec3(parent_rotation, local_bone_positions(i)) + parent_position;
            global_bone_rotations(i) = quat_mul(parent_rotation, local_bone_rotations(i));
        }
    }
}

// Compute forward kinematics of just some joints using a
// mask to indicate which joints are already computed
void forward_kinematics_partial(
    slice1d<vec3> global_bone_positions,
    slice1d<quat> global_bone_rotations,
    slice1d<bool> global_bone_computed,
    const slice1d<vec3> local_bone_positions,
    const slice1d<quat> local_bone_rotations,
    const slice1d<int> bone_parents,
    int bone)
{
    if (bone_parents(bone) == -1)
    {
        global_bone_positions(bone) = local_bone_positions(bone);
        global_bone_rotations(bone) = local_bone_rotations(bone);
        global_bone_computed(bone) = true;
        return;
    }
    
    if (!global_bone_computed(bone_parents(bone)))
    {
        forward_kinematics_partial(
            global_bone_positions,
            global_bone_rotations,
            global_bone_computed,
            local_bone_positions,
            local_bone_rotations,
            bone_parents,
            bone_parents(bone));
    }
    
    vec3 parent_position = global_bone_positions(bone_parents(bone));
    quat parent_rotation = global_bone_rotations(bone_parents(bone));
    global_bone_positions(bone) = quat_mul_vec3(parent_rotation, local_bone_positions(bone)) + parent_position;
    global_bone_rotations(bone) = quat_mul(parent_rotation, local_bone_rotations(bone));
    global_bone_computed(bone) = true;
}

// Same but including velocity
void forward_kinematics_velocity_partial(
    slice1d<vec3> global_bone_positions,
    slice1d<vec3> global_bone_velocities,
    slice1d<quat> global_bone_rotations,
    slice1d<vec3> global_bone_angular_velocities,
    slice1d<bool> global_bone_computed,
    const slice1d<vec3> local_bone_positions,
    const slice1d<vec3> local_bone_velocities,
    const slice1d<quat> local_bone_rotations,
    const slice1d<vec3> local_bone_angular_velocities,
    const slice1d<int> bone_parents,
    int bone)
{
    if (bone_parents(bone) == -1)
    {
        global_bone_positions(bone) = local_bone_positions(bone);
        global_bone_velocities(bone) = local_bone_velocities(bone);
        global_bone_rotations(bone) = local_bone_rotations(bone);
        global_bone_angular_velocities(bone) = local_bone_angular_velocities(bone);
        global_bone_computed(bone) = true;
        return;
    }
    
    if (!global_bone_computed(bone_parents(bone)))
    {
        forward_kinematics_velocity_partial(
            global_bone_positions,
            global_bone_velocities,
            global_bone_rotations,
            global_bone_angular_velocities,
            global_bone_computed,
            local_bone_positions,
            local_bone_velocities,
            local_bone_rotations,
            local_bone_angular_velocities,
            bone_parents,
            bone_parents(bone));
    }
    
    vec3 parent_position = global_bone_positions(bone_parents(bone));
    vec3 parent_velocity = global_bone_velocities(bone_parents(bone));
    quat parent_rotation = global_bone_rotations(bone_parents(bone));
    vec3 parent_angular_velocity = global_bone_angular_velocities(bone_parents(bone));
    
    global_bone_positions(bone) = quat_mul_vec3(parent_rotation, local_bone_positions(bone)) + parent_position;
    global_bone_velocities(bone) = 
        parent_velocity + 
        quat_mul_vec3(parent_rotation, local_bone_velocities(bone)) + 
        cross(parent_angular_velocity, quat_mul_vec3(parent_rotation, local_bone_positions(bone)));
    global_bone_rotations(bone) = quat_mul(parent_rotation, local_bone_rotations(bone));
    global_bone_angular_velocities(bone) = quat_mul_vec3(parent_rotation, local_bone_angular_velocities(bone)) + parent_angular_velocity;
    global_bone_computed(bone) = true;
}

//--------------------------------------

// Compute a feature for the position of a bone relative to the simulation/root bone
void compute_bone_position_feature(database& db, int& offset, int bone, float weight = 1.0f)
{
    for (int i = 0; i < db.nframes(); i++)
    {
        vec3 bone_position;
        quat bone_rotation;
        
        forward_kinematics(
            bone_position,
            bone_rotation,
            db.bone_positions(i),
            db.bone_rotations(i),
            db.bone_parents,
            bone);
        
        bone_position = quat_mul_vec3(quat_inv(db.bone_rotations(i, 0)), bone_position - db.bone_positions(i, 0));
        
        db.features(i, offset + 0) = bone_position.x;
        db.features(i, offset + 1) = bone_position.y;
        db.features(i, offset + 2) = bone_position.z;
    }
    
    normalize_feature(db.features, db.features_offset, db.features_scale, offset, 3, weight);
    
    offset += 3;
}

// Similar but for a bone's velocity
void compute_bone_velocity_feature(database& db, int& offset, int bone, float weight = 1.0f)
{
    for (int i = 0; i < db.nframes(); i++)
    {
        vec3 bone_position;
        vec3 bone_velocity;
        quat bone_rotation;
        vec3 bone_angular_velocity;
        
        forward_kinematics_velocity(
            bone_position,
            bone_velocity,
            bone_rotation,
            bone_angular_velocity,
            db.bone_positions(i),
            db.bone_velocities(i),
            db.bone_rotations(i),
            db.bone_angular_velocities(i),
            db.bone_parents,
            bone);
        
        bone_velocity = quat_mul_vec3(quat_inv(db.bone_rotations(i, 0)), bone_velocity);
        
        db.features(i, offset + 0) = bone_velocity.x;
        db.features(i, offset + 1) = bone_velocity.y;
        db.features(i, offset + 2) = bone_velocity.z;
    }
    
    normalize_feature(db.features, db.features_offset, db.features_scale, offset, 3, weight);
    
    offset += 3;
}

static inline void database_trajectory_horizons(
    int out[3], const float fps)
{
    assert(isfinite(fps) && fps > 0.0f);
    out[0] = int(roundf(fps / 3.0f));
    out[1] = int(roundf(2.0f * fps / 3.0f));
    out[2] = int(roundf(fps));
}

// Compute the trajectory at one-third, two-thirds, and one second in the future
void compute_trajectory_position_feature(
    database& db, int& offset, const float fps, float weight = 1.0f)
{
    int horizons[3];
    database_trajectory_horizons(horizons, fps);

    for (int i = 0; i < db.nframes(); i++)
    {
        int t0 = database_trajectory_index_clamp(db, i, horizons[0]);
        int t1 = database_trajectory_index_clamp(db, i, horizons[1]);
        int t2 = database_trajectory_index_clamp(db, i, horizons[2]);
        
        vec3 trajectory_pos0 = quat_mul_vec3(quat_inv(db.bone_rotations(i, 0)), db.bone_positions(t0, 0) - db.bone_positions(i, 0));
        vec3 trajectory_pos1 = quat_mul_vec3(quat_inv(db.bone_rotations(i, 0)), db.bone_positions(t1, 0) - db.bone_positions(i, 0));
        vec3 trajectory_pos2 = quat_mul_vec3(quat_inv(db.bone_rotations(i, 0)), db.bone_positions(t2, 0) - db.bone_positions(i, 0));
        
        db.features(i, offset + 0) = trajectory_pos0.x;
        db.features(i, offset + 1) = trajectory_pos0.z;
        db.features(i, offset + 2) = trajectory_pos1.x;
        db.features(i, offset + 3) = trajectory_pos1.z;
        db.features(i, offset + 4) = trajectory_pos2.x;
        db.features(i, offset + 5) = trajectory_pos2.z;
    }
    
    normalize_feature(db.features, db.features_offset, db.features_scale, offset, 6, weight);
    
    offset += 6;
}

// Same for direction
void compute_trajectory_direction_feature(
    database& db, int& offset, const float fps, float weight = 1.0f)
{
    int horizons[3];
    database_trajectory_horizons(horizons, fps);

    for (int i = 0; i < db.nframes(); i++)
    {
        int t0 = database_trajectory_index_clamp(db, i, horizons[0]);
        int t1 = database_trajectory_index_clamp(db, i, horizons[1]);
        int t2 = database_trajectory_index_clamp(db, i, horizons[2]);
        
        vec3 trajectory_dir0 = quat_mul_vec3(quat_inv(db.bone_rotations(i, 0)), quat_mul_vec3(db.bone_rotations(t0, 0), vec3(0, 0, 1)));
        vec3 trajectory_dir1 = quat_mul_vec3(quat_inv(db.bone_rotations(i, 0)), quat_mul_vec3(db.bone_rotations(t1, 0), vec3(0, 0, 1)));
        vec3 trajectory_dir2 = quat_mul_vec3(quat_inv(db.bone_rotations(i, 0)), quat_mul_vec3(db.bone_rotations(t2, 0), vec3(0, 0, 1)));
        
        db.features(i, offset + 0) = trajectory_dir0.x;
        db.features(i, offset + 1) = trajectory_dir0.z;
        db.features(i, offset + 2) = trajectory_dir1.x;
        db.features(i, offset + 3) = trajectory_dir1.z;
        db.features(i, offset + 4) = trajectory_dir2.x;
        db.features(i, offset + 5) = trajectory_dir2.z;
    }

    normalize_feature(db.features, db.features_offset, db.features_scale, offset, 6, weight);

    offset += 6;
}

static inline void compute_terrain_feature(
    database& db, int& offset, const float weight)
{
    assert(db.terrain_features.rows == db.nframes());
    assert(db.terrain_features.cols == 4);

    for (int i = 0; i < db.nframes(); ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            db.features(i, offset + j) = db.terrain_features(i, j);
        }
    }

    normalize_feature(
        db.features, db.features_offset, db.features_scale, offset, 4, weight);
    offset += 4;
}

static inline float database_frame_cost(
    const database& db, const int frame, const slice1d<float> query)
{
    assert(frame >= 0 && frame < db.nframes());
    assert(query.size == db.nfeatures());

    float cost = 0.0f;
    for (int i = 0; i < db.nfeatures(); ++i)
    {
        const float normalized = normalize_query_feature(
            query(i), db.features_offset(i), db.features_scale(i));
        cost += squaref(normalized - db.features(frame, i));
    }
    return cost;
}

static inline float database_raw_terrain_error(
    const database& db, const int frame, const slice1d<float> query)
{
    assert(frame >= 0 && frame < db.nframes());
    assert(query.size >= 31);
    assert(db.terrain_features.rows == db.nframes());
    assert(db.terrain_features.cols == 4);

    float error = 0.0f;
    for (int j = 0; j < 4; ++j)
    {
        error += squaref(query(27 + j) - db.terrain_features(frame, j));
    }
    return error;
}

// Build the Motion Matching search acceleration structure. Here we
// just use axis aligned bounding boxes regularly spaced at BOUND_SM_SIZE
// and BOUND_LR_SIZE frames
void database_build_bounds(database& db)
{
    int nbound_sm = ((db.nframes() + BOUND_SM_SIZE - 1) / BOUND_SM_SIZE);
    int nbound_lr = ((db.nframes() + BOUND_LR_SIZE - 1) / BOUND_LR_SIZE);
    
    db.bound_sm_min.resize(nbound_sm, db.nfeatures()); 
    db.bound_sm_max.resize(nbound_sm, db.nfeatures()); 
    db.bound_lr_min.resize(nbound_lr, db.nfeatures()); 
    db.bound_lr_max.resize(nbound_lr, db.nfeatures()); 
    
    db.bound_sm_min.set(+FLT_MAX);
    db.bound_sm_max.set(-FLT_MAX);
    db.bound_lr_min.set(+FLT_MAX);
    db.bound_lr_max.set(-FLT_MAX);
    
    for (int i = 0; i < db.nframes(); i++)
    {
        int i_sm = i / BOUND_SM_SIZE;
        int i_lr = i / BOUND_LR_SIZE;
        
        for (int j = 0; j < db.nfeatures(); j++)
        {
            db.bound_sm_min(i_sm, j) = minf(db.bound_sm_min(i_sm, j), db.features(i, j));
            db.bound_sm_max(i_sm, j) = maxf(db.bound_sm_max(i_sm, j), db.features(i, j));
            db.bound_lr_min(i_lr, j) = minf(db.bound_lr_min(i_lr, j), db.features(i, j));
            db.bound_lr_max(i_lr, j) = maxf(db.bound_lr_max(i_lr, j), db.features(i, j));
        }
    }
}

// Build all motion matching features and acceleration structure
void database_build_matching_features(
    database& db,
    const float feature_weight_foot_position,
    const float feature_weight_foot_velocity,
    const float feature_weight_hip_velocity,
    const float feature_weight_trajectory_positions,
    const float feature_weight_trajectory_directions,
    const int left_foot_bone,
    const int right_foot_bone,
    const int hip_bone,
    const float feature_weight_terrain = 0.0f,
    const float fps = 60.0f)
{
    if (left_foot_bone < 0 || left_foot_bone >= db.nbones() ||
        right_foot_bone < 0 || right_foot_bone >= db.nbones() ||
        hip_bone < 0 || hip_bone >= db.nbones())
    {
        return;
    }

    const float feature_weights[6] = {
        feature_weight_foot_position,
        feature_weight_foot_velocity,
        feature_weight_hip_velocity,
        feature_weight_trajectory_positions,
        feature_weight_trajectory_directions,
        feature_weight_terrain
    };
    for (int i = 0; i < 6; ++i)
    {
        if (!feature_weight_is_valid(feature_weights[i]))
        {
            return;
        }
    }

    if (!feature_float_is_positive_finite(fps) ||
        db.terrain_features.rows != db.nframes() ||
        db.terrain_features.cols != 4)
    {
        return;
    }

    int nfeatures = 
        3 + // Left Foot Position
        3 + // Right Foot Position 
        3 + // Left Foot Velocity
        3 + // Right Foot Velocity
        3 + // Hip Velocity
        6 + // Trajectory Positions 2D
        6 + // Trajectory Directions 2D
        4 ; // Terrain Centerline Heights
        
    db.features.resize(db.nframes(), nfeatures);
    db.features_offset.resize(nfeatures);
    db.features_scale.resize(nfeatures);
    
    int offset = 0;
    compute_bone_position_feature(db, offset, left_foot_bone, feature_weight_foot_position);
    compute_bone_position_feature(db, offset, right_foot_bone, feature_weight_foot_position);
    compute_bone_velocity_feature(db, offset, left_foot_bone, feature_weight_foot_velocity);
    compute_bone_velocity_feature(db, offset, right_foot_bone, feature_weight_foot_velocity);
    compute_bone_velocity_feature(db, offset, hip_bone, feature_weight_hip_velocity);
    compute_trajectory_position_feature(
        db, offset, fps, feature_weight_trajectory_positions);
    compute_trajectory_direction_feature(
        db, offset, fps, feature_weight_trajectory_directions);
    compute_terrain_feature(db, offset, feature_weight_terrain);
    
    assert(offset == nfeatures);
    
    database_build_bounds(db);
}

enum database_candidate_verdict
{
    DatabaseCandidateAccept,
    DatabaseCandidateReject,
    DatabaseCandidateFatal
};

enum database_search_status
{
    DatabaseSearchComplete,
    DatabaseSearchInvalidInput,
    DatabaseSearchCandidateFatal
};

struct database_candidate_validator
{
    void* context = nullptr;
    database_candidate_verdict (*evaluate)(void*, int) = nullptr;
};

constexpr int DatabaseUseIncumbentNeighborhood = -2;

// Motion Matching search function essentially consists
// of comparing every feature vector in the database,
// against the query feature vector, first checking the
// query distance to the axis aligned bounding boxes used
// for the acceleration structure.
database_search_status motion_matching_search_validated(
    int& __restrict__ best_index,
    float& __restrict__ best_cost,
    const slice1d<int> range_starts,
    const slice1d<int> range_stops,
    const slice2d<float> features,
    const slice1d<float> features_offset,
    const slice1d<float> features_scale,
    const slice2d<float> bound_sm_min,
    const slice2d<float> bound_sm_max,
    const slice2d<float> bound_lr_min,
    const slice2d<float> bound_lr_max,
    const slice1d<float> query_normalized,
    const float transition_cost,
    const int ignore_range_end,
    const int ignore_surrounding,
    const unsigned char* candidate_mask = nullptr,
    const int candidate_mask_count = 0,
    const int neighborhood_center = DatabaseUseIncumbentNeighborhood,
    const database_candidate_validator* validator = nullptr)
{
    // Keep strict header builds warning-clean while these legacy public API
    // parameters remain unused by the normalized-distance implementation.
    (void)features_offset;
    (void)features_scale;

    const int nfeatures = query_normalized.size;
    const int nranges = range_starts.size;
    const int resolved_neighborhood_center =
        neighborhood_center == DatabaseUseIncumbentNeighborhood
            ? best_index
            : neighborhood_center;

    if (best_index < -1 || best_index >= features.rows ||
        resolved_neighborhood_center < -1 ||
        resolved_neighborhood_center >= features.rows ||
        ignore_range_end < 0 || ignore_surrounding < 0 ||
        (candidate_mask != nullptr &&
         candidate_mask_count != features.rows) ||
        (validator != nullptr && validator->evaluate == nullptr))
    {
        best_index = -1;
        best_cost = FLT_MAX;
        return DatabaseSearchInvalidInput;
    }

    const int curr_index = resolved_neighborhood_center;

    if (candidate_mask != nullptr)
    {
        if (best_index != -1 && candidate_mask[best_index] != 1)
        {
            best_index = -1;
            best_cost = FLT_MAX;
        }
    }
    
    // Find cost for current frame
    if (best_index != -1)
    {
        best_cost = 0.0;
        for (int i = 0; i < nfeatures; i++)
        {
            best_cost += squaref(query_normalized(i) - features(best_index, i));
        }
    }
    
    float curr_cost = 0.0f;
    
    // Search rest of database
    for (int r = 0; r < nranges; r++)
    {
        // Exclude end of ranges from search    
        int i = range_starts(r);
        int range_end = range_stops(r) - ignore_range_end;
        
        while (i < range_end)
        {
            // Find index of current and next large box
            int i_lr = i / BOUND_LR_SIZE;
            int i_lr_next = (i_lr + 1) * BOUND_LR_SIZE;
            
            // Find distance to box
            curr_cost = transition_cost;
            for (int j = 0; j < nfeatures; j++)
            {
                curr_cost += squaref(query_normalized(j) - clampf(query_normalized(j), 
                    bound_lr_min(i_lr, j), bound_lr_max(i_lr, j)));
                
                if (curr_cost >= best_cost)
                {
                    break;
                }
            }
            
            // If distance is greater than current best jump to next box
            if (curr_cost >= best_cost)
            {
                i = i_lr_next;
                continue;
            }
            
            // Check against small box
            while (i < i_lr_next && i < range_end)
            {   
                // Find index of current and next small box
                int i_sm = i / BOUND_SM_SIZE;
                int i_sm_next = (i_sm + 1) * BOUND_SM_SIZE;
                
                // Find distance to box
                curr_cost = transition_cost;
                for (int j = 0; j < nfeatures; j++)
                {
                    curr_cost += squaref(query_normalized(j) - clampf(query_normalized(j), 
                        bound_sm_min(i_sm, j), bound_sm_max(i_sm, j)));
                    
                    if (curr_cost >= best_cost)
                    {
                        break;
                    }
                }
                
                // If distance is greater than current best jump to next box
                if (curr_cost >= best_cost)
                {
                    i = i_sm_next;
                    continue;
                }
                
                // Search inside small box
                while (i < i_sm_next && i < range_end)
                {
                    if (candidate_mask != nullptr && candidate_mask[i] != 1)
                    {
                        i++;
                        continue;
                    }

                    // Skip surrounding frames
                    if (curr_index != - 1 && abs(i - curr_index) < ignore_surrounding)
                    {
                        i++;
                        continue;
                    }
                    
                    // Check against each frame inside small box
                    curr_cost = transition_cost;
                    for (int j = 0; j < nfeatures; j++)
                    {
                        curr_cost += squaref(query_normalized(j) - features(i, j));
                        if (curr_cost >= best_cost)
                        {
                            break;
                        }
                    }
                    
                    // Validate only candidates that can beat the lowest-cost
                    // accepted incumbent. A rejected candidate must not lower
                    // the pruning threshold.
                    if (curr_cost < best_cost)
                    {
                        const database_candidate_verdict verdict =
                            validator == nullptr
                                ? DatabaseCandidateAccept
                                : validator->evaluate(validator->context, i);
                        if (verdict == DatabaseCandidateAccept)
                        {
                            best_index = i;
                            best_cost = curr_cost;
                        }
                        else if (verdict != DatabaseCandidateReject)
                        {
                            best_index = -1;
                            best_cost = FLT_MAX;
                            return DatabaseSearchCandidateFatal;
                        }
                    }
                    
                    i++;
                }
            }
        }
    }

    return DatabaseSearchComplete;
}

void motion_matching_search(
    int& __restrict__ best_index,
    float& __restrict__ best_cost,
    const slice1d<int> range_starts,
    const slice1d<int> range_stops,
    const slice2d<float> features,
    const slice1d<float> features_offset,
    const slice1d<float> features_scale,
    const slice2d<float> bound_sm_min,
    const slice2d<float> bound_sm_max,
    const slice2d<float> bound_lr_min,
    const slice2d<float> bound_lr_max,
    const slice1d<float> query_normalized,
    const float transition_cost,
    const int ignore_range_end,
    const int ignore_surrounding,
    const unsigned char* candidate_mask = nullptr,
    const int candidate_mask_count = 0)
{
    // Preserve the legacy malformed-mask result exactly.
    if (candidate_mask != nullptr && candidate_mask_count != features.rows)
    {
        best_index = -1;
        return;
    }
    (void)motion_matching_search_validated(
        best_index,
        best_cost,
        range_starts,
        range_stops,
        features,
        features_offset,
        features_scale,
        bound_sm_min,
        bound_sm_max,
        bound_lr_min,
        bound_lr_max,
        query_normalized,
        transition_cost,
        ignore_range_end,
        ignore_surrounding,
        candidate_mask,
        candidate_mask_count,
        DatabaseUseIncumbentNeighborhood,
        nullptr);
}

static bool database_search_shape_is_valid(const database& db)
{
    const int frames = db.nframes();
    const int features = db.nfeatures();
    const int expected_small_bounds =
        (frames + BOUND_SM_SIZE - 1) / BOUND_SM_SIZE;
    const int expected_large_bounds =
        (frames + BOUND_LR_SIZE - 1) / BOUND_LR_SIZE;
    if (frames <= 0 || db.bone_positions.cols <= 0 ||
        db.bone_positions.data == nullptr || features <= 0 ||
        db.features.rows != frames || db.features.data == nullptr ||
        db.features_offset.size != features ||
        db.features_offset.data == nullptr ||
        db.features_scale.size != features ||
        db.features_scale.data == nullptr ||
        db.range_starts.size <= 0 ||
        db.range_starts.size != db.range_stops.size ||
        db.range_starts.data == nullptr || db.range_stops.data == nullptr ||
        db.bound_sm_min.rows != expected_small_bounds ||
        db.bound_sm_min.cols != features || db.bound_sm_min.data == nullptr ||
        db.bound_sm_max.rows != expected_small_bounds ||
        db.bound_sm_max.cols != features || db.bound_sm_max.data == nullptr ||
        db.bound_lr_min.rows != expected_large_bounds ||
        db.bound_lr_min.cols != features || db.bound_lr_min.data == nullptr ||
        db.bound_lr_max.rows != expected_large_bounds ||
        db.bound_lr_max.cols != features || db.bound_lr_max.data == nullptr)
    {
        return false;
    }

    int prior_stop = 0;
    for (int range = 0; range < db.nranges(); ++range)
    {
        const int start = db.range_starts(range);
        const int stop = db.range_stops(range);
        if (start < prior_stop || start < 0 || stop <= start || stop > frames)
        {
            return false;
        }
        prior_stop = stop;
    }
    return true;
}

database_search_status database_search_validated(
    int& best_index,
    float& best_cost,
    const database& db,
    const slice1d<float> query,
    const float transition_cost,
    const int ignore_range_end,
    const int ignore_surrounding,
    const unsigned char* candidate_mask,
    const int candidate_mask_count,
    const int neighborhood_center,
    const database_candidate_validator* validator)
{
    if (!database_search_shape_is_valid(db) ||
        query.size != db.nfeatures() || query.data == nullptr ||
        best_index < -1 || best_index >= db.nframes() ||
        (neighborhood_center != DatabaseUseIncumbentNeighborhood &&
         (neighborhood_center < -1 || neighborhood_center >= db.nframes())) ||
        ignore_range_end < 0 || ignore_surrounding < 0 ||
        (candidate_mask != nullptr &&
         candidate_mask_count != db.nframes()) ||
        (validator != nullptr && validator->evaluate == nullptr))
    {
        best_index = -1;
        best_cost = FLT_MAX;
        return DatabaseSearchInvalidInput;
    }

    if (best_index == -1)
    {
        best_cost = FLT_MAX;
    }

    array1d<float> query_normalized(db.nfeatures());
    for (int i = 0; i < db.nfeatures(); i++)
    {
        query_normalized(i) = normalize_query_feature(
            query(i), db.features_offset(i), db.features_scale(i));
    }

    return motion_matching_search_validated(
        best_index,
        best_cost,
        db.range_starts,
        db.range_stops,
        db.features,
        db.features_offset,
        db.features_scale,
        db.bound_sm_min,
        db.bound_sm_max,
        db.bound_lr_min,
        db.bound_lr_max,
        query_normalized,
        transition_cost,
        ignore_range_end,
        ignore_surrounding,
        candidate_mask,
        candidate_mask_count,
        neighborhood_center,
        validator);
}

// Search database
void database_search(
    int& best_index, 
    float& best_cost, 
    const database& db, 
    const slice1d<float> query,
    const float transition_cost = 0.0f,
    const int ignore_range_end = 20,
    const int ignore_surrounding = 20,
    const unsigned char* candidate_mask = nullptr,
    const int candidate_mask_count = 0)
{
    if (candidate_mask != nullptr && candidate_mask_count != db.nframes())
    {
        best_index = -1;
        return;
    }

    // Normalize Query
    array1d<float> query_normalized(db.nfeatures());
    for (int i = 0; i < db.nfeatures(); i++)
    {
        query_normalized(i) = normalize_query_feature(
            query(i), db.features_offset(i), db.features_scale(i));
    }
    
    // Search
    motion_matching_search(
        best_index, 
        best_cost, 
        db.range_starts,
        db.range_stops,
        db.features,
        db.features_offset,
        db.features_scale,
        db.bound_sm_min,
        db.bound_sm_max,
        db.bound_lr_min,
        db.bound_lr_max,
        query_normalized,
        transition_cost,
        ignore_range_end,
        ignore_surrounding,
        candidate_mask,
        candidate_mask_count);
}
