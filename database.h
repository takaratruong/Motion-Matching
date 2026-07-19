#pragma once

#include "common.h"
#include "vec.h"
#include "quat.h"
#include "array.h"
#include "motion_index_runtime.h"

#include <assert.h>
#include <float.h>
#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

//--------------------------------------

enum
{
    BOUND_SM_SIZE = 16,
    BOUND_LR_SIZE = 64,
    DATABASE_POSE_TRAJECTORY_FEATURES = 27,
    DATABASE_LEGACY_TERRAIN_FEATURES = 4,
    DATABASE_G1_TERRAIN_FEATURES = 12,
    DATABASE_LEGACY_MATCHING_FEATURES = 31,
    DATABASE_G1_MATCHING_FEATURES = 39,
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

// When we add an offset to a frame in the database there is a chance
// it will go out of the relevant range so here we can clamp it to 
// the last frame of that range.
static inline int database_range_index_for_frame(
    const database& db, int frame, int* probe_count = NULL)
{
    int probes = 0;
    int low = 0;
    int high = db.nranges();
    while (low < high)
    {
        ++probes;
        const int middle = low + (high - low) / 2;
        if (frame < db.range_starts(middle))
        {
            high = middle;
        }
        else if (frame >= db.range_stops(middle))
        {
            low = middle + 1;
        }
        else
        {
            if (probe_count != NULL) *probe_count = probes;
            return middle;
        }
    }
    if (probe_count != NULL) *probe_count = probes;
    return -1;
}

int database_trajectory_index_clamp(const database& db, int frame, int offset)
{
    const int range = database_range_index_for_frame(db, frame);
    if (range >= 0)
    {
        return clamp(frame + offset, db.range_starts(range), db.range_stops(range) - 1);
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

static inline void database_trajectory_horizons(int out[3])
{
    out[0] = 8;
    out[1] = 17;
    out[2] = 25;
}

// Compute the trajectory at one-third, two-thirds, and one second in the future
void compute_trajectory_position_feature(database& db, int& offset, float weight = 1.0f)
{
    int horizons[3];
    database_trajectory_horizons(horizons);

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
void compute_trajectory_direction_feature(database& db, int& offset, float weight = 1.0f)
{
    int horizons[3];
    database_trajectory_horizons(horizons);

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
    assert(db.terrain_features.cols == DATABASE_LEGACY_TERRAIN_FEATURES ||
           db.terrain_features.cols == DATABASE_G1_TERRAIN_FEATURES);

    for (int i = 0; i < db.nframes(); ++i)
    {
        for (int j = 0; j < db.terrain_features.cols; ++j)
        {
            db.features(i, offset + j) = db.terrain_features(i, j);
        }
    }

    normalize_feature(
        db.features, db.features_offset, db.features_scale, offset,
        db.terrain_features.cols, weight);
    offset += db.terrain_features.cols;
}

static inline bool database_matching_feature_contract_is_valid(
    const database& db, const int expected_terrain_dimensions)
{
    if (expected_terrain_dimensions != DATABASE_LEGACY_TERRAIN_FEATURES &&
        expected_terrain_dimensions != DATABASE_G1_TERRAIN_FEATURES)
        return false;
    const int expected_features =
        DATABASE_POSE_TRAJECTORY_FEATURES + expected_terrain_dimensions;
    return db.terrain_features.rows == db.nframes() &&
           db.terrain_features.cols == expected_terrain_dimensions &&
           db.features.rows == db.nframes() &&
           db.features.cols == expected_features &&
           db.features_offset.size == expected_features &&
           db.features_scale.size == expected_features;
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
    assert(query.size >= DATABASE_POSE_TRAJECTORY_FEATURES +
                         db.terrain_features.cols);
    assert(db.terrain_features.rows == db.nframes());
    assert(db.terrain_features.cols == DATABASE_LEGACY_TERRAIN_FEATURES ||
           db.terrain_features.cols == DATABASE_G1_TERRAIN_FEATURES);

    float error = 0.0f;
    for (int j = 0; j < db.terrain_features.cols; ++j)
    {
        error += squaref(
            query(DATABASE_POSE_TRAJECTORY_FEATURES + j) -
            db.terrain_features(frame, j));
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
    const float feature_weight_terrain = 0.0f)
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

    if (db.terrain_features.rows != db.nframes() ||
        (db.terrain_features.cols != DATABASE_LEGACY_TERRAIN_FEATURES &&
         db.terrain_features.cols != DATABASE_G1_TERRAIN_FEATURES))
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
        db.terrain_features.cols; // Terrain profile/corridor descriptor
        
    db.features.resize(db.nframes(), nfeatures);
    db.features_offset.resize(nfeatures);
    db.features_scale.resize(nfeatures);
    
    int offset = 0;
    compute_bone_position_feature(db, offset, left_foot_bone, feature_weight_foot_position);
    compute_bone_position_feature(db, offset, right_foot_bone, feature_weight_foot_position);
    compute_bone_velocity_feature(db, offset, left_foot_bone, feature_weight_foot_velocity);
    compute_bone_velocity_feature(db, offset, right_foot_bone, feature_weight_foot_velocity);
    compute_bone_velocity_feature(db, offset, hip_bone, feature_weight_hip_velocity);
    compute_trajectory_position_feature(db, offset, feature_weight_trajectory_positions);
    compute_trajectory_direction_feature(db, offset, feature_weight_trajectory_directions);
    compute_terrain_feature(db, offset, feature_weight_terrain);
    
    assert(offset == nfeatures);
    
    database_build_bounds(db);
}

// Motion Matching search function essentially consists
// of comparing every feature vector in the database, 
// against the query feature vector, first checking the 
// query distance to the axis aligned bounding boxes used 
// for the acceleration structure.
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
    const int ignore_surrounding)
{
    // Keep strict header builds warning-clean while these legacy public API
    // parameters remain unused by the normalized-distance implementation.
    (void)features_offset;
    (void)features_scale;

    int nfeatures = query_normalized.size;
    int nranges = range_starts.size;
    
    int curr_index = best_index;
    
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
                    
                    // If cost is lower than current best then update best
                    if (curr_cost < best_cost)
                    {
                        best_index = i;
                        best_cost = curr_cost;
                    }
                    
                    i++;
                }
            }
        }
    }
}

// Search database
void database_search(
    int& best_index, 
    float& best_cost, 
    const database& db, 
    const slice1d<float> query,
    const float transition_cost = 0.0f,
    const int ignore_range_end = 20,
    const int ignore_surrounding = 20)
{
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
        ignore_surrounding);
}

//--------------------------------------
// Terrain-bank and direction-compatible indexed search.

enum database_indexed_search_status
{
    DATABASE_INDEXED_SEARCH_INVALID = -1,
    DATABASE_INDEXED_SEARCH_EMPTY = 0,
    DATABASE_INDEXED_SEARCH_FOUND = 1,
};

struct database_indexed_search_result
{
    database_indexed_search_status status = DATABASE_INDEXED_SEARCH_EMPTY;
    int index = -1;
    float cost = FLT_MAX;
    uint64_t eligible_frame_count = 0;
    uint64_t evaluated_frame_count = 0;
    uint64_t considered_bound_count = 0;
    uint64_t skipped_bound_count = 0;
};

static inline bool database_indexed_range_contract_is_valid(const database& db)
{
    if (db.nranges() <= 0 || db.range_stops.size != db.nranges()) return false;
    int cursor = 0;
    for (int range = 0; range < db.nranges(); ++range)
    {
        if (db.range_starts(range) != cursor ||
            db.range_stops(range) <= db.range_starts(range) ||
            db.range_stops(range) > db.nframes())
            return false;
        cursor = db.range_stops(range);
    }
    return cursor == db.nframes();
}

static inline bool database_indexed_storage_is_valid(
    const database& db, const motion_index_runtime& index)
{
    if (db.nframes() <= 0 ||
        !database_matching_feature_contract_is_valid(
            db, db.terrain_features.cols) ||
        !database_indexed_range_contract_is_valid(db))
        return false;

    const int small_bounds =
        (db.nframes() + BOUND_SM_SIZE - 1) / BOUND_SM_SIZE;
    const int large_bounds =
        (db.nframes() + BOUND_LR_SIZE - 1) / BOUND_LR_SIZE;
    if (db.bound_sm_min.rows != small_bounds ||
        db.bound_sm_max.rows != small_bounds ||
        db.bound_lr_min.rows != large_bounds ||
        db.bound_lr_max.rows != large_bounds ||
        db.bound_sm_min.cols != db.nfeatures() ||
        db.bound_sm_max.cols != db.nfeatures() ||
        db.bound_lr_min.cols != db.nfeatures() ||
        db.bound_lr_max.cols != db.nfeatures())
        return false;

    const size_t frames = static_cast<size_t>(db.nframes());
    return index.direction_masks.size() == frames &&
           index.speed_masks.size() == frames &&
           index.elevation_modes.size() == frames &&
           index.range_direction_masks.size() ==
               static_cast<size_t>(db.nranges()) &&
           index.range_speed_masks.size() ==
               static_cast<size_t>(db.nranges()) &&
           index.range_elevation_masks.size() ==
               static_cast<size_t>(db.nranges()) &&
           index.small_bound_direction_masks.size() ==
               static_cast<size_t>(small_bounds) &&
           index.small_bound_speed_masks.size() ==
               static_cast<size_t>(small_bounds) &&
           index.small_bound_elevation_masks.size() ==
               static_cast<size_t>(small_bounds) &&
           index.large_bound_direction_masks.size() ==
               static_cast<size_t>(large_bounds) &&
           index.large_bound_speed_masks.size() ==
               static_cast<size_t>(large_bounds) &&
           index.large_bound_elevation_masks.size() ==
               static_cast<size_t>(large_bounds) &&
           index.small_bound_size == BOUND_SM_SIZE &&
           index.large_bound_size == BOUND_LR_SIZE;
}

static inline bool database_indexed_selected_ranges_are_valid(
    const database& db, const int* ranges, int range_count)
{
    if (range_count < 0 || (range_count > 0 && ranges == NULL)) return false;
    int prior = -1;
    for (int selected = 0; selected < range_count; ++selected)
    {
        if (ranges[selected] <= prior || ranges[selected] >= db.nranges())
            return false;
        prior = ranges[selected];
    }
    return true;
}

static inline bool database_indexed_query_is_valid(
    const database& db, const slice1d<float> query)
{
    if (query.size != db.nfeatures() ||
        (query.size > 0 && query.data == NULL)) return false;
    for (int dimension = 0; dimension < db.nfeatures(); ++dimension)
    {
        if (!feature_float_is_finite(db.features_offset(dimension))) return false;
        const float scale = db.features_scale(dimension);
        if (!feature_scale_is_disabled(scale) &&
            !feature_float_is_positive_finite(scale))
            return false;
        if (!feature_scale_is_disabled(scale) &&
            !feature_float_is_finite(query(dimension)))
            return false;
        const float normalized = normalize_query_feature(
            query(dimension), db.features_offset(dimension), scale);
        if (!feature_float_is_finite(normalized)) return false;
    }
    return true;
}

static inline bool database_indexed_range_is_selected(
    const int* ranges, int range_count, int sought)
{
    int low = 0;
    int high = range_count;
    while (low < high)
    {
        const int middle = low + (high - low) / 2;
        if (ranges[middle] < sought) low = middle + 1;
        else high = middle;
    }
    return low < range_count && ranges[low] == sought;
}

static inline bool database_indexed_frame_has_compatible_published_horizon(
    const database& db,
    const motion_index_runtime& index,
    int frame,
    uint16_t direction_mask,
    uint8_t speed_mask,
    int elevation_mode,
    int minimum_future_published_frames)
{
    if (minimum_future_published_frames < 1)
        return false;
    const int range = database_range_index_for_frame(db, frame);
    if (range < 0 ||
        minimum_future_published_frames >
            db.range_stops(range) - 1 - frame)
        return false;
    for (int offset = 0;
         offset <= minimum_future_published_frames;
         ++offset)
    {
        if (!motion_index_row_is_compatible(
                index, static_cast<size_t>(frame + offset), direction_mask,
                speed_mask, elevation_mode))
            return false;
    }
    return true;
}

static inline bool database_indexed_frame_is_publishable(
    const database& db,
    const motion_index_runtime& index,
    int frame,
    uint16_t direction_mask,
    uint8_t speed_mask,
    int elevation_mode)
{
    return database_indexed_frame_has_compatible_published_horizon(
        db, index, frame, direction_mask, speed_mask, elevation_mode, 1);
}

static inline int database_indexed_search_range_end(
    const database& db, int range, int ignore_range_end)
{
    const int start = db.range_starts(range);
    const int stop = db.range_stops(range);
    return ignore_range_end >= stop - start ? start : stop - ignore_range_end;
}

static inline int database_indexed_min_int(int first, int second)
{
    return first < second ? first : second;
}

static inline void database_indexed_count_eligible(
    database_indexed_search_result& result,
    const database& db,
    const motion_index_runtime& index,
    const int* ranges,
    int range_count,
    uint16_t direction_mask,
    uint8_t speed_mask,
    int elevation_mode,
    int incumbent_frame,
    int incumbent_range,
    int ignore_range_end,
    int ignore_surrounding,
    int minimum_future_published_frames)
{
    for (int selected = 0; selected < range_count; ++selected)
    {
        const int range = ranges[selected];
        ++result.considered_bound_count;
        if (!motion_index_aggregate_is_compatible(
                index.range_direction_masks[static_cast<size_t>(range)],
                index.range_speed_masks[static_cast<size_t>(range)],
                index.range_elevation_masks[static_cast<size_t>(range)],
                direction_mask, speed_mask, elevation_mode))
        {
            ++result.skipped_bound_count;
            continue;
        }

        int frame = db.range_starts(range);
        const int range_end = database_indexed_search_range_end(
            db, range, ignore_range_end);
        while (frame < range_end)
        {
            const int large = frame / BOUND_LR_SIZE;
            const int large_end = database_indexed_min_int(
                (large + 1) * BOUND_LR_SIZE, range_end);
            ++result.considered_bound_count;
            if (!motion_index_aggregate_is_compatible(
                    index.large_bound_direction_masks[
                        static_cast<size_t>(large)],
                    index.large_bound_speed_masks[static_cast<size_t>(large)],
                    index.large_bound_elevation_masks[
                        static_cast<size_t>(large)],
                    direction_mask, speed_mask, elevation_mode))
            {
                ++result.skipped_bound_count;
                frame = large_end;
                continue;
            }

            while (frame < large_end)
            {
                const int small = frame / BOUND_SM_SIZE;
                const int small_end = database_indexed_min_int(
                    (small + 1) * BOUND_SM_SIZE, large_end);
                ++result.considered_bound_count;
                if (!motion_index_aggregate_is_compatible(
                        index.small_bound_direction_masks[
                            static_cast<size_t>(small)],
                        index.small_bound_speed_masks[
                            static_cast<size_t>(small)],
                        index.small_bound_elevation_masks[
                            static_cast<size_t>(small)],
                        direction_mask, speed_mask, elevation_mode))
                {
                    ++result.skipped_bound_count;
                    frame = small_end;
                    continue;
                }
                while (frame < small_end)
                {
                    const bool surrounding =
                        incumbent_frame >= 0 && range == incumbent_range &&
                        abs(frame - incumbent_frame) < ignore_surrounding;
                    if (!surrounding &&
                        database_indexed_frame_has_compatible_published_horizon(
                            db, index, frame, direction_mask,
                            speed_mask, elevation_mode,
                            minimum_future_published_frames))
                        ++result.eligible_frame_count;
                    ++frame;
                }
            }
        }
    }
}

static inline database_indexed_search_status database_search_indexed(
    database_indexed_search_result& output,
    const database& db,
    const motion_index_runtime& index,
    const int* compatible_range_indices,
    int compatible_range_count,
    uint16_t direction_mask,
    uint8_t speed_mask,
    int elevation_mode,
    const slice1d<float> query,
    int incumbent_frame,
    float transition_cost = 0.0f,
    int ignore_range_end = 20,
    int ignore_surrounding = 20,
    int minimum_future_published_frames = 1)
{
    if (!database_indexed_storage_is_valid(db, index) ||
        !database_indexed_selected_ranges_are_valid(
            db, compatible_range_indices, compatible_range_count) ||
        !motion_index_direction_is_valid(direction_mask) ||
        !motion_index_speed_is_valid(speed_mask) ||
        !motion_index_elevation_is_valid(elevation_mode) ||
        !database_indexed_query_is_valid(db, query) ||
        incumbent_frame < -1 || incumbent_frame >= db.nframes() ||
        !feature_weight_is_valid(transition_cost) ||
        ignore_range_end < 0 || ignore_surrounding < 0 ||
        minimum_future_published_frames < 1)
        return DATABASE_INDEXED_SEARCH_INVALID;

    array1d<float> query_normalized;
    query_normalized.resize(db.nfeatures());
    for (int dimension = 0; dimension < db.nfeatures(); ++dimension)
        query_normalized(dimension) = normalize_query_feature(
            query(dimension), db.features_offset(dimension),
            db.features_scale(dimension));

    database_indexed_search_result candidate;
    const int incumbent_range = incumbent_frame >= 0
        ? database_range_index_for_frame(db, incumbent_frame) : -1;
    const bool incumbent_compatible =
        incumbent_frame >= 0 && incumbent_range >= 0 &&
        database_indexed_range_is_selected(
            compatible_range_indices, compatible_range_count, incumbent_range) &&
        database_indexed_frame_is_publishable(
            db, index, incumbent_frame, direction_mask,
            speed_mask, elevation_mode);
    if (incumbent_compatible)
    {
        candidate.status = DATABASE_INDEXED_SEARCH_FOUND;
        candidate.index = incumbent_frame;
        candidate.cost = database_frame_cost(db, incumbent_frame, query);
        ++candidate.evaluated_frame_count;
        if (!feature_float_is_finite(candidate.cost))
            return DATABASE_INDEXED_SEARCH_INVALID;
    }

    database_indexed_count_eligible(
        candidate, db, index, compatible_range_indices,
        compatible_range_count, direction_mask, speed_mask, elevation_mode,
        incumbent_frame, incumbent_range, ignore_range_end, ignore_surrounding,
        minimum_future_published_frames);
    if (candidate.eligible_frame_count == 0)
    {
        output = candidate;
        return candidate.status;
    }

    for (int selected = 0; selected < compatible_range_count; ++selected)
    {
        const int range = compatible_range_indices[selected];
        if (!motion_index_aggregate_is_compatible(
                index.range_direction_masks[static_cast<size_t>(range)],
                index.range_speed_masks[static_cast<size_t>(range)],
                index.range_elevation_masks[static_cast<size_t>(range)],
                direction_mask, speed_mask, elevation_mode))
            continue;

        int frame = db.range_starts(range);
        const int range_end = database_indexed_search_range_end(
            db, range, ignore_range_end);
        while (frame < range_end)
        {
            const int large = frame / BOUND_LR_SIZE;
            const int large_end = database_indexed_min_int(
                (large + 1) * BOUND_LR_SIZE, range_end);
            if (!motion_index_aggregate_is_compatible(
                    index.large_bound_direction_masks[
                        static_cast<size_t>(large)],
                    index.large_bound_speed_masks[static_cast<size_t>(large)],
                    index.large_bound_elevation_masks[
                        static_cast<size_t>(large)],
                    direction_mask, speed_mask, elevation_mode))
            {
                frame = large_end;
                continue;
            }

            float bound_cost = transition_cost;
            for (int dimension = 0; dimension < db.nfeatures(); ++dimension)
            {
                bound_cost += squaref(
                    query_normalized(dimension) -
                    clampf(query_normalized(dimension),
                           db.bound_lr_min(large, dimension),
                           db.bound_lr_max(large, dimension)));
                if (bound_cost >= candidate.cost) break;
            }
            if (!feature_float_is_finite(bound_cost))
                return DATABASE_INDEXED_SEARCH_INVALID;
            if (bound_cost >= candidate.cost)
            {
                frame = large_end;
                continue;
            }

            while (frame < large_end)
            {
                const int small = frame / BOUND_SM_SIZE;
                const int small_end = database_indexed_min_int(
                    (small + 1) * BOUND_SM_SIZE, large_end);
                if (!motion_index_aggregate_is_compatible(
                        index.small_bound_direction_masks[
                            static_cast<size_t>(small)],
                        index.small_bound_speed_masks[
                            static_cast<size_t>(small)],
                        index.small_bound_elevation_masks[
                            static_cast<size_t>(small)],
                        direction_mask, speed_mask, elevation_mode))
                {
                    frame = small_end;
                    continue;
                }

                bound_cost = transition_cost;
                for (int dimension = 0; dimension < db.nfeatures(); ++dimension)
                {
                    bound_cost += squaref(
                        query_normalized(dimension) -
                        clampf(query_normalized(dimension),
                               db.bound_sm_min(small, dimension),
                               db.bound_sm_max(small, dimension)));
                    if (bound_cost >= candidate.cost) break;
                }
                if (!feature_float_is_finite(bound_cost))
                    return DATABASE_INDEXED_SEARCH_INVALID;
                if (bound_cost >= candidate.cost)
                {
                    frame = small_end;
                    continue;
                }

                while (frame < small_end)
                {
                    const bool surrounding =
                        incumbent_frame >= 0 && range == incumbent_range &&
                        abs(frame - incumbent_frame) < ignore_surrounding;
                    if (surrounding ||
                        !database_indexed_frame_has_compatible_published_horizon(
                            db, index, frame, direction_mask,
                            speed_mask, elevation_mode,
                            minimum_future_published_frames))
                    {
                        ++frame;
                        continue;
                    }

                    ++candidate.evaluated_frame_count;
                    float cost = transition_cost;
                    for (int dimension = 0; dimension < db.nfeatures(); ++dimension)
                    {
                        cost += squaref(
                            query_normalized(dimension) -
                            db.features(frame, dimension));
                        if (cost >= candidate.cost) break;
                    }
                    if (!feature_float_is_finite(cost))
                        return DATABASE_INDEXED_SEARCH_INVALID;
                    if (cost < candidate.cost)
                    {
                        candidate.status = DATABASE_INDEXED_SEARCH_FOUND;
                        candidate.index = frame;
                        candidate.cost = cost;
                    }
                    ++frame;
                }
            }
        }
    }

    if (candidate.status != DATABASE_INDEXED_SEARCH_FOUND)
        return DATABASE_INDEXED_SEARCH_INVALID;
    output = candidate;
    return candidate.status;
}
