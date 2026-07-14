#pragma once

#include "g1_controller_state.h"

struct scene_model_load_result
{
    bool allocated = false;
    bool ready = false;
};

template<class Model, class SceneLoader, class ModelLoader, class Unloader>
static inline bool scene_switch_transaction(
    scene_pack& active_scene,
    g1_controller_state& active_state,
    Model& active_model,
    int& active_index,
    int target_index,
    const database& db,
    const terrain_support_set& support,
    SceneLoader load_scene,
    ModelLoader load_model,
    Unloader unload,
    char* error,
    int capacity)
{
    scene_pack candidate_scene;
    if (!load_scene(candidate_scene, target_index, error, capacity))
    {
        return false;
    }

    g1_controller_state candidate_state;
    if (!g1_controller_state_reset(
            candidate_state,
            db,
            support,
            candidate_scene,
            error,
            capacity))
    {
        return false;
    }

    Model candidate_model = {};
    const scene_model_load_result loaded = load_model(
        candidate_model,
        candidate_scene.mesh_path.c_str(),
        error,
        capacity);
    if (!loaded.ready)
    {
        if (loaded.allocated)
        {
            unload(candidate_model);
        }
        return false;
    }

    Model old_model = active_model;
    active_model = candidate_model;
    scene_pack_swap(active_scene, candidate_scene);
    g1_controller_state_swap(active_state, candidate_state);
    active_index = target_index;
    unload(old_model);
    return true;
}

static inline bool scene_reset_current(
    g1_controller_state& state,
    const database& db,
    const terrain_support_set& support,
    const scene_pack& scene,
    char* error,
    int capacity)
{
    g1_controller_state candidate;
    if (!g1_controller_state_reset(
            candidate, db, support, scene, error, capacity))
    {
        return false;
    }
    g1_controller_state_swap(state, candidate);
    return true;
}
