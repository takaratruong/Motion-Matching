#pragma once

#include "interaction_ik.h"
#include "interaction_place_target.h"

#include <cstdint>
#include <vector>

namespace interaction {

class PlaceController;

enum class PlaceMotionMode : uint8_t {
    None,
    RecordedPlace,
    ReversedPickup,
};

enum class PlacePhase : uint8_t {
    Align,
    Lower,
    Release,
    Retract,
    Finished,
};

struct PlaceTimingConfig {
    float canonical_fps = 25.0F;
    float playback_speed = 1.0F;
    float entry_blend_seconds = 0.25F;
    float reversed_commit_seconds = 0.50F;
    float maximum_alignment_seconds = 1.00F;
};

struct PlaceMatchConfig {
    float maximum_entry_root_error_m = 0.25F;
    float maximum_entry_yaw_error_radians = 0.436332313F;
};

struct RecordedPlaceClip {
    uint64_t id = 0;
    uint64_t object_profile_id = 0;
    uint32_t fps_numerator = 25;
    uint32_t fps_denominator = 1;
    std::vector<Pose> poses;
    std::vector<Transform> object_poses;
    std::vector<uint8_t> active_hand_contacts;
    int32_t entry_frame = -1;
    int32_t commit_frame = -1;
    int32_t release_frame = -1;
    int32_t retract_stop_frame = -1;
    Hand hand = Hand::Right;
    Transform hand_in_object{};
    ObjectLocalBounds object_bounds{};
    PlacementSurface source_surface{};
    uint32_t source_affordance_id = 0;
};

struct PlaceMotionLibrary {
    std::vector<RecordedPlaceClip> recorded;
};

struct PlaceMatchInput {
    const Database* pickup_database = nullptr;
    const PlaceMotionLibrary* library = nullptr;
    TargetHandle held_target{};
    MatchCandidate pickup_candidate{};
    Pose current_pose{};
    Transform current_object_world{};
    uint64_t held_object_profile_id = 0;
    ObjectLocalBounds held_object_bounds{};
    GraspAffordance held_affordance{};
    PlacementSurface surface{};
    PlaceAffordance place_affordance{};
    vec3 object_dimensions{};
    PlaceTimingConfig timing{};
    PlaceMatchConfig match{};
    IKConfig ik{};
};

struct PlaceCandidate {
    PlaceMotionMode mode = PlaceMotionMode::None;
    uint64_t source_id = 0;
    uint64_t selection_id = 0;
    PlaceTimingConfig timing{};
    PlaceMatchConfig match{};
    IKConfig ik{};
    int32_t clip = -1;
    int32_t entry_frame = -1;
    int32_t commit_frame = -1;
    int32_t release_frame = -1;
    int32_t stop_frame = -1;
    int32_t direction = 0;
    Transform scene_from_source{};
    Transform staging_root_world{};
    vec3 entry_root_offset{};
    float entry_yaw_offset = 0.0F;
    float total_cost = 0.0F;
};

struct PlaceResult {
    bool accepted = false;
    PlaceCandidate candidate{};
    Reason reason = Reason::None;
};

struct PlaceSample {
    Pose pose{};
    Transform source_object{};
    int32_t source_frame = -1;
    PlacePhase phase = PlacePhase::Align;
    bool committed = false;
};

struct PlaceStagingPreview {
    bool accepted = false;
    bool ready = false;
    Reason reason = Reason::None;
    PlaceCandidate candidate{};
    IKConfig ik{};
    uint64_t ik_config_fingerprint = 0;
    Transform staging_root_world{};
    float root_error_m = 0.0F;
    float yaw_error_radians = 0.0F;
};

PlaceResult select_place_motion(const PlaceMatchInput& input);
PlaceStagingPreview preview_place_motion(const PlaceMatchInput& input);

class PlacePlayer {
public:
    void start(
        const PlaceCandidate& candidate,
        const PlaceMatchInput& input);
    void advance(float dt);
    PlaceSample sample() const;
    int32_t source_frame() const;
    double source_frame_exact() const;
    PlacePhase phase() const;
    bool committed() const;
    bool release_due() const;
    void acknowledge_release();
    bool finished() const;

private:
    friend class PlaceController;
    void start_validated(
        const PlaceCandidate& candidate,
        const PlaceMatchInput& input);
    rotation_gate::Rotation mapped_root_rotation_evidence() const;
    PlaceCandidate candidate_{};
    PlaceMatchInput input_{};
    double source_frame_ = 0.0;
    int32_t segment_tick_ = 0;
    int32_t commit_tick_count_ = 0;
    int32_t release_tick_count_ = 0;
    int32_t stop_tick_count_ = 0;
    bool started_ = false;
    bool committed_ = false;
    bool release_latched_ = false;
    bool release_acknowledged_ = false;
    bool finished_ = true;
};

}  // namespace interaction
