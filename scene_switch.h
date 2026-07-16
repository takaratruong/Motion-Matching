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
    g1_controller_state_memory_range ranges[5][64] = {};
    int counts[5] = {};
    if (error_capacity < 0 ||
        (extra_object_0 == nullptr) != (extra_object_0_bytes == 0U) ||
        (extra_object_1 == nullptr) != (extra_object_1_bytes == 0U) ||
        !g1_frame_runtime_storage_sets_are_safe(
            runtime, false, ranges, counts)) {
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
        for (int state_index = 0; state_index < 5; ++state_index) {
            if (g1_frame_ranges_overlap_object(
                    ranges[state_index],
                    counts[state_index],
                    objects[first].data,
                    objects[first].bytes)) {
                return false;
            }
        }
    }

    const g1_controller_state_memory_range diagnostic = {
        error,
        error != nullptr && error_capacity > 0
            ? static_cast<std::size_t>(error_capacity)
            : 0U
    };
    if (diagnostic.bytes > 0U &&
        g1_controller_state_source_storage_overlaps(
            diagnostic, db, support, scene)) {
        return false;
    }
    for (int state_index = 0; state_index < 5; ++state_index) {
        if (g1_frame_error_overlaps_ranges(
                error,
                error_capacity,
                ranges[state_index],
                counts[state_index]) ||
            g1_frame_ranges_overlap_sources(
                ranges[state_index],
                counts[state_index],
                db,
                support,
                scene)) {
            return false;
        }
    }
    return true;
}

static inline bool scene_frame_runtime_reset_candidate_is_valid(
    const G1FrameRuntime& runtime,
    const G1FrameResetConfig& config)
{
    const g1_controller_state* states[5] = {};
    g1_frame_runtime_state_pointers(runtime, states);
    for (int state_index = 0; state_index < 5; ++state_index) {
        if (!g1_frame_reset_candidate_is_valid(
                *states[state_index], config.initial_search_time)) {
            return false;
        }
    }
    return g1_frame_runtime_states_are_logically_equal(runtime) &&
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
    g1_controller_state_memory_range live_ranges[5][64] = {};
    g1_controller_state_memory_range candidate_ranges[5][64] = {};
    int live_counts[5] = {};
    int candidate_counts[5] = {};
    if (error_capacity < 0 ||
        (extra_object_0 == nullptr) != (extra_object_0_bytes == 0U) ||
        (extra_object_1 == nullptr) != (extra_object_1_bytes == 0U) ||
        !g1_frame_runtime_storage_sets_are_safe(
            live_runtime, false, live_ranges, live_counts) ||
        !g1_frame_runtime_storage_sets_are_safe(
            candidate_runtime,
            true,
            candidate_ranges,
            candidate_counts)) {
        return false;
    }

    const g1_controller_state* live_states[5] = {};
    const g1_controller_state* candidate_states[5] = {};
    g1_frame_runtime_state_pointers(live_runtime, live_states);
    g1_frame_runtime_state_pointers(candidate_runtime, candidate_states);
    const g1_controller_state* states[10] = {
        live_states[0], live_states[1], live_states[2],
        live_states[3], live_states[4],
        candidate_states[0], candidate_states[1], candidate_states[2],
        candidate_states[3], candidate_states[4],
    };
    const g1_controller_state_memory_range* range_sets[10] = {
        live_ranges[0], live_ranges[1], live_ranges[2],
        live_ranges[3], live_ranges[4],
        candidate_ranges[0], candidate_ranges[1], candidate_ranges[2],
        candidate_ranges[3], candidate_ranges[4],
    };
    const int range_counts[10] = {
        live_counts[0], live_counts[1], live_counts[2],
        live_counts[3], live_counts[4],
        candidate_counts[0], candidate_counts[1], candidate_counts[2],
        candidate_counts[3], candidate_counts[4],
    };
    for (int first = 0; first < 10; ++first) {
        for (int second = first + 1; second < 10; ++second) {
            if (states[first] == states[second] ||
                !g1_frame_state_range_sets_are_disjoint(
                    *states[first],
                    range_sets[first],
                    range_counts[first],
                    *states[second],
                    range_sets[second],
                    range_counts[second])) {
                return false;
            }
        }
    }

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
        for (int set = 0; set < 10; ++set) {
            if (g1_frame_ranges_overlap_object(
                    range_sets[set],
                    range_counts[set],
                    objects[first].data,
                    objects[first].bytes)) {
                return false;
            }
        }
    }

    for (int set = 0; set < 10; ++set) {
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
    g1_controller_state_swap(
        first.candidates.common_state,
        second.candidates.common_state);
    g1_controller_state_swap(
        first.candidates.raw_state,
        second.candidates.raw_state);
    g1_controller_state_swap(
        first.candidates.ik_state,
        second.candidates.ik_state);
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
