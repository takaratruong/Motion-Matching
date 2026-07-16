#pragma once

#include "g1_frame_transaction.h"

#include <cstddef>
#include <type_traits>
#include <utility>

struct scene_model_load_result
{
    bool allocated = false;
    bool ready = false;
};

static_assert(
    std::is_nothrow_swappable<G1FramePublication>::value,
    "frame publication must have a non-throwing commit swap");
static_assert(
    std::is_nothrow_swappable<G1FrameAcceptedDiagnostic>::value,
    "accepted diagnostics must have a non-throwing commit swap");

static inline bool scene_frame_runtime_live_storage_preflight(
    const G1FrameRuntime& runtime,
    const database& db,
    const terrain_support_set& support,
    const scene_pack& scene,
    const G1FrameResetConfig& config,
    const void* extra_object_0,
    std::size_t extra_object_0_bytes,
    const void* extra_object_1,
    std::size_t extra_object_1_bytes,
    char* error,
    int error_capacity)
{
    g1_controller_state_memory_range accepted_ranges[64] = {};
    g1_controller_state_memory_range working_ranges[64] = {};
    int accepted_count = 0;
    int working_count = 0;
    if (error_capacity < 0 ||
        (extra_object_0 == nullptr) != (extra_object_0_bytes == 0U) ||
        (extra_object_1 == nullptr) != (extra_object_1_bytes == 0U) ||
        !g1_frame_state_pair_reset_storage_is_safe(
            runtime.accepted_state,
            runtime.working_state,
            accepted_ranges,
            accepted_count,
            working_ranges,
            working_count,
            64)) {
        return false;
    }

    const g1_controller_state_memory_range objects[] = {
        {&runtime, sizeof(runtime)},
        {&config, sizeof(config)},
        {extra_object_0, extra_object_0_bytes},
        {extra_object_1, extra_object_1_bytes},
    };
    const int object_count = static_cast<int>(
        sizeof(objects) / sizeof(objects[0]));
    for (int first = 0; first < object_count; ++first) {
        if (objects[first].bytes == 0U) continue;
        if (g1_controller_state_source_storage_overlaps(
                objects[first], db, support, scene)) {
            return false;
        }
        for (int second = first + 1;
             second < object_count;
             ++second) {
            if (objects[second].bytes != 0U &&
                g1_controller_state_ranges_overlap(
                    objects[first], objects[second])) {
                return false;
            }
        }
        if (g1_frame_error_overlaps_object(
                error,
                error_capacity,
                objects[first].data,
                objects[first].bytes)) {
            return false;
        }
        if (g1_frame_ranges_overlap_object(
                accepted_ranges,
                accepted_count,
                objects[first].data,
                objects[first].bytes) ||
            g1_frame_ranges_overlap_object(
                working_ranges,
                working_count,
                objects[first].data,
                objects[first].bytes)) {
            return false;
        }
    }

    const g1_controller_state_memory_range diagnostic = {
        error,
        error != nullptr && error_capacity > 0
            ? static_cast<std::size_t>(error_capacity)
            : 0U
    };
    return !g1_frame_error_overlaps_ranges(
               error,
               error_capacity,
               accepted_ranges,
               accepted_count) &&
           !g1_frame_error_overlaps_ranges(
               error,
               error_capacity,
               working_ranges,
               working_count) &&
           (diagnostic.bytes == 0U ||
            !g1_controller_state_source_storage_overlaps(
                diagnostic, db, support, scene)) &&
           !g1_frame_ranges_overlap_sources(
               accepted_ranges,
               accepted_count,
               db,
               support,
               scene) &&
           !g1_frame_ranges_overlap_sources(
               working_ranges,
               working_count,
               db,
               support,
               scene);
}

static inline bool scene_frame_runtime_reset_candidate_is_valid(
    const G1FrameRuntime& runtime,
    const G1FrameResetConfig& config)
{
    return g1_frame_reset_candidate_is_valid(
               runtime.accepted_state,
               config.initial_search_time) &&
           g1_frame_reset_candidate_is_valid(
               runtime.working_state,
               config.initial_search_time) &&
           g1_frame_controller_states_equal(
               runtime.accepted_state,
               runtime.working_state) &&
           g1_frame_publication_is_valid(runtime.publication) &&
           !runtime.publication.rejection.rejected &&
           !runtime.publication.ik_safe_stop_latched &&
           runtime.publication.presentation_frame == 0 &&
           g1_frame_intent_bits_equal(
               runtime.publication.requested_intent,
               runtime.accepted_state.command.intent) &&
           g1_frame_accepted_diagnostic_is_valid(
               runtime.accepted_diagnostic) &&
           !runtime.accepted_diagnostic.ready &&
           g1_frame_runtime_observation_relation_is_valid(runtime);
}

static inline bool scene_frame_runtime_candidate_is_isolated(
    const G1FrameRuntime& live_runtime,
    const G1FrameRuntime& candidate_runtime,
    const database& db,
    const terrain_support_set& support,
    const scene_pack& live_scene,
    const scene_pack& candidate_scene,
    const G1FrameResetConfig& config,
    const void* extra_object_0,
    std::size_t extra_object_0_bytes,
    const void* extra_object_1,
    std::size_t extra_object_1_bytes,
    char* error,
    int error_capacity)
{
    g1_controller_state_memory_range live_accepted_ranges[64] = {};
    g1_controller_state_memory_range live_working_ranges[64] = {};
    g1_controller_state_memory_range candidate_accepted_ranges[64] = {};
    g1_controller_state_memory_range candidate_working_ranges[64] = {};
    int live_accepted_count = 0;
    int live_working_count = 0;
    int candidate_accepted_count = 0;
    int candidate_working_count = 0;
    if (error_capacity < 0 ||
        !g1_frame_state_pair_reset_storage_is_safe(
            live_runtime.accepted_state,
            live_runtime.working_state,
            live_accepted_ranges,
            live_accepted_count,
            live_working_ranges,
            live_working_count,
            64) ||
        !g1_frame_state_pair_storage_is_exact(
            candidate_runtime.accepted_state,
            candidate_runtime.working_state,
            candidate_accepted_ranges,
            candidate_accepted_count,
            candidate_working_ranges,
            candidate_working_count,
            64)) {
        return false;
    }

    const bool state_sets_are_disjoint =
        g1_frame_state_range_sets_are_disjoint(
            live_runtime.accepted_state,
            live_accepted_ranges,
            live_accepted_count,
            candidate_runtime.accepted_state,
            candidate_accepted_ranges,
            candidate_accepted_count) &&
        g1_frame_state_range_sets_are_disjoint(
            live_runtime.accepted_state,
            live_accepted_ranges,
            live_accepted_count,
            candidate_runtime.working_state,
            candidate_working_ranges,
            candidate_working_count) &&
        g1_frame_state_range_sets_are_disjoint(
            live_runtime.working_state,
            live_working_ranges,
            live_working_count,
            candidate_runtime.accepted_state,
            candidate_accepted_ranges,
            candidate_accepted_count) &&
        g1_frame_state_range_sets_are_disjoint(
            live_runtime.working_state,
            live_working_ranges,
            live_working_count,
            candidate_runtime.working_state,
            candidate_working_ranges,
            candidate_working_count);
    if (!state_sets_are_disjoint) return false;

    const g1_controller_state_memory_range objects[] = {
        {&live_runtime, sizeof(live_runtime)},
        {&candidate_runtime, sizeof(candidate_runtime)},
        {&config, sizeof(config)},
        {extra_object_0, extra_object_0_bytes},
        {extra_object_1, extra_object_1_bytes},
    };
    const int object_count = static_cast<int>(
        sizeof(objects) / sizeof(objects[0]));
    for (int first = 0; first < object_count; ++first) {
        if (objects[first].bytes == 0U) continue;
        if (g1_controller_state_source_storage_overlaps(
                objects[first], db, support, live_scene) ||
            g1_controller_state_source_storage_overlaps(
                objects[first], db, support, candidate_scene)) {
            return false;
        }
        if (g1_frame_error_overlaps_object(
                error,
                error_capacity,
                objects[first].data,
                objects[first].bytes)) {
            return false;
        }
        for (int second = first + 1;
             second < object_count;
             ++second) {
            if (objects[second].bytes != 0U &&
                g1_controller_state_ranges_overlap(
                    objects[first], objects[second])) {
                return false;
            }
        }
        if (g1_frame_ranges_overlap_object(
                live_accepted_ranges,
                live_accepted_count,
                objects[first].data,
                objects[first].bytes) ||
            g1_frame_ranges_overlap_object(
                live_working_ranges,
                live_working_count,
                objects[first].data,
                objects[first].bytes) ||
            g1_frame_ranges_overlap_object(
                candidate_accepted_ranges,
                candidate_accepted_count,
                objects[first].data,
                objects[first].bytes) ||
            g1_frame_ranges_overlap_object(
                candidate_working_ranges,
                candidate_working_count,
                objects[first].data,
                objects[first].bytes)) {
            return false;
        }
    }

    const g1_controller_state_memory_range* const range_sets[] = {
        live_accepted_ranges,
        live_working_ranges,
        candidate_accepted_ranges,
        candidate_working_ranges,
    };
    const int range_counts[] = {
        live_accepted_count,
        live_working_count,
        candidate_accepted_count,
        candidate_working_count,
    };
    for (int set = 0; set < 4; ++set) {
        if (g1_frame_error_overlaps_ranges(
                error,
                error_capacity,
                range_sets[set],
                range_counts[set]) ||
            g1_frame_ranges_overlap_sources(
                range_sets[set],
                range_counts[set],
                db,
                support,
                live_scene) ||
            g1_frame_ranges_overlap_sources(
                range_sets[set],
                range_counts[set],
                db,
                support,
                candidate_scene)) {
            return false;
        }
    }
    const g1_controller_state_memory_range diagnostic = {
        error,
        error != nullptr && error_capacity > 0
            ? static_cast<std::size_t>(error_capacity)
            : 0U
    };
    return diagnostic.bytes == 0U ||
           (!g1_controller_state_source_storage_overlaps(
                diagnostic, db, support, live_scene) &&
            !g1_controller_state_source_storage_overlaps(
                diagnostic, db, support, candidate_scene));
}

static inline void scene_frame_runtime_swap(
    G1FrameRuntime& first,
    G1FrameRuntime& second) noexcept
{
    g1_controller_state_swap(
        first.accepted_state, second.accepted_state);
    g1_controller_state_swap(
        first.working_state, second.working_state);
    using std::swap;
    swap(first.publication, second.publication);
    swap(first.accepted_diagnostic, second.accepted_diagnostic);
}

template<class Model, class SceneLoader, class ModelLoader, class Unloader>
static inline bool scene_switch_transaction(
    scene_pack& active_scene,
    G1FrameRuntime& active_runtime,
    Model& active_model,
    int& active_index,
    int target_index,
    const database& db,
    const terrain_support_set& support,
    const G1FrameResetConfig& config,
    SceneLoader load_scene,
    ModelLoader load_model,
    Unloader unload,
    char* error,
    int capacity)
{
    static_assert(
        std::is_nothrow_swappable<Model>::value,
        "the active scene model must have a non-throwing commit swap");
    if (!scene_frame_runtime_live_storage_preflight(
            active_runtime,
            db,
            support,
            active_scene,
            config,
            &active_model,
            sizeof(active_model),
            &active_index,
            sizeof(active_index),
            error,
            capacity)) {
        return false;
    }

    scene_pack candidate_scene;
    if (!load_scene(candidate_scene, target_index, error, capacity)) {
        return false;
    }

    G1FrameRuntime candidate_runtime;
    if (!g1_frame_runtime_reset(
            candidate_runtime,
            db,
            support,
            candidate_scene,
            config,
            error,
            capacity)) {
        return false;
    }
    if (!scene_frame_runtime_reset_candidate_is_valid(
            candidate_runtime, config) ||
        !scene_frame_runtime_candidate_is_isolated(
            active_runtime,
            candidate_runtime,
            db,
            support,
            active_scene,
            candidate_scene,
            config,
            &active_model,
            sizeof(active_model),
            &active_index,
            sizeof(active_index),
            error,
            capacity)) {
        return scene_error(
            error,
            capacity,
            "scene switch: candidate frame runtime is invalid");
    }

    Model candidate_model = {};
    const scene_model_load_result loaded = load_model(
        candidate_model,
        candidate_scene.mesh_path.c_str(),
        error,
        capacity);
    if (!loaded.ready || !loaded.allocated) {
        if (loaded.allocated) {
            unload(candidate_model);
        }
        if (loaded.ready) {
            return scene_error(
                error,
                capacity,
                "scene switch: ready candidate model was not allocated");
        }
        return false;
    }

    // All validation and every potentially failing load precedes this tail.
    // The swaps install the complete live unit before the old model is freed.
    scene_pack_swap(active_scene, candidate_scene);
    scene_frame_runtime_swap(active_runtime, candidate_runtime);
    using std::swap;
    swap(active_model, candidate_model);
    swap(active_index, target_index);
    unload(candidate_model);
    return true;
}

static inline bool scene_reset_current(
    G1FrameRuntime& runtime,
    const database& db,
    const terrain_support_set& support,
    const scene_pack& scene,
    const G1FrameResetConfig& config,
    char* error,
    int capacity)
{
    if (!scene_frame_runtime_live_storage_preflight(
            runtime,
            db,
            support,
            scene,
            config,
            nullptr,
            0U,
            nullptr,
            0U,
            error,
            capacity)) {
        return false;
    }

    G1FrameRuntime candidate;
    if (!g1_frame_runtime_reset(
            candidate,
            db,
            support,
            scene,
            config,
            error,
            capacity)) {
        return false;
    }
    if (!scene_frame_runtime_reset_candidate_is_valid(candidate, config) ||
        !scene_frame_runtime_candidate_is_isolated(
            runtime,
            candidate,
            db,
            support,
            scene,
            scene,
            config,
            nullptr,
            0U,
            nullptr,
            0U,
            error,
            capacity)) {
        return scene_error(
            error,
            capacity,
            "scene reset: candidate frame runtime is invalid");
    }

    scene_frame_runtime_swap(runtime, candidate);
    return true;
}
