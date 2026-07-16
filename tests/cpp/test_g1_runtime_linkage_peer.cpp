// The legacy database header also contains out-of-line definitions. Rename
// those pre-existing globals in this peer translation unit so this regression
// isolates the helpers newly owned by the public runtime header.
#define database_load peer_database_load
#define database_save_matching_features peer_database_save_matching_features
#define database_trajectory_index_clamp peer_database_trajectory_index_clamp
#define normalize_feature peer_normalize_feature
#define denormalize_features peer_denormalize_features
#define forward_kinematics peer_forward_kinematics
#define forward_kinematics_velocity peer_forward_kinematics_velocity
#define forward_kinematics_full peer_forward_kinematics_full
#define forward_kinematics_partial peer_forward_kinematics_partial
#define forward_kinematics_velocity_partial \
    peer_forward_kinematics_velocity_partial
#define compute_bone_position_feature peer_compute_bone_position_feature
#define compute_bone_velocity_feature peer_compute_bone_velocity_feature
#define compute_trajectory_position_feature \
    peer_compute_trajectory_position_feature
#define compute_trajectory_direction_feature \
    peer_compute_trajectory_direction_feature
#define database_build_bounds peer_database_build_bounds
#define database_build_matching_features peer_database_build_matching_features
#define motion_matching_search peer_motion_matching_search
#define database_search peer_database_search

#include "sonic/cpp/g1_runtime.h"

#undef database_load
#undef database_save_matching_features
#undef database_trajectory_index_clamp
#undef normalize_feature
#undef denormalize_features
#undef forward_kinematics
#undef forward_kinematics_velocity
#undef forward_kinematics_full
#undef forward_kinematics_partial
#undef forward_kinematics_velocity_partial
#undef compute_bone_position_feature
#undef compute_bone_velocity_feature
#undef compute_trajectory_position_feature
#undef compute_trajectory_direction_feature
#undef database_build_bounds
#undef database_build_matching_features
#undef motion_matching_search
#undef database_search

int g1_runtime_linkage_peer_mode()
{
    g1_runtime_step_request request;
    return request.mode;
}
