#include "episode_grasp_provider.h"
#include "episode_controls.h"
#include "episode_reach_planner.h"
#include "g1_mesh_renderer.h"
#include "interaction_episode.h"
#include "reach_search.h"
#include "raylib.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <future>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

constexpr float kPi = 3.14159265358979323846F;
constexpr float kTableTop = 0.65F;
constexpr vec3 kTableDimensions(1.20F, 0.06F, 0.75F);
constexpr vec3 kObjectDimensions(0.10F, 0.10F, 0.10F);

struct Options {
    std::filesystem::path episode_pack{"build/g1-episode"};
    std::filesystem::path reach_pack{
        "build/g1-reaches/reach-pack-v2"};
};

struct OrbitCamera {
    vec3 target{0.0F, 0.75F, 0.0F};
    float azimuth = -0.80F;
    float altitude = 0.32F;
    float distance = 3.70F;
};

enum class SearchPurpose : uint8_t { Pickup, Place };

struct SearchJob {
    SearchPurpose purpose = SearchPurpose::Pickup;
    episode::ObjectSnapshot object{};
    episode::GraspCandidate grasp{};
    reach::ExhaustiveQuery query{};
    interaction::OrientedBox object_box{};
    interaction::EnvironmentGeometry environment{};
    interaction::PlacedSupportContext support{};
    std::optional<reach::Hand> required_hand{};
    std::shared_ptr<std::atomic_bool> cancellation;
    std::future<reach::SearchResult> result;
};

Vector3 ray(vec3 value) {
    return {value.x, value.y, value.z};
}

Options parse_options(int argc, char** argv) {
    Options options{};
    if (argc > 3) {
        throw std::invalid_argument(
            "usage: g1_interaction_episode_viewer [EPISODE_PACK] "
            "[REACH_PACK]");
    }
    if (argc >= 2) options.episode_pack = argv[1];
    if (argc >= 3) options.reach_pack = argv[2];
    return options;
}

void update_camera(Camera3D& camera, OrbitCamera& orbit) {
    const Vector2 mouse = GetMouseDelta();
    if (IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
        orbit.azimuth = std::remainder(
            orbit.azimuth - 0.006F * mouse.x, 2.0F * kPi);
        orbit.altitude = std::clamp(
            orbit.altitude + 0.006F * mouse.y, -1.30F, 1.30F);
    }
    orbit.distance = std::clamp(
        orbit.distance * std::exp(-0.16F * GetMouseWheelMove()),
        0.70F,
        9.0F);

    const float cosine = std::cos(orbit.altitude);
    const vec3 offset(
        cosine * std::cos(orbit.azimuth),
        std::sin(orbit.altitude),
        cosine * std::sin(orbit.azimuth));
    const vec3 forward = -offset;
    const vec3 up(0.0F, 1.0F, 0.0F);
    const vec3 right = normalize(cross(forward, up));
    const vec3 view_up = normalize(cross(right, forward));
    if (IsMouseButtonDown(MOUSE_BUTTON_MIDDLE)) {
        const float scale = 0.0015F * orbit.distance;
        orbit.target = orbit.target +
            scale * (mouse.x * right + mouse.y * view_up);
    }
    const vec3 position = orbit.target + orbit.distance * offset;
    camera.position = ray(position);
    camera.target = ray(orbit.target);
    camera.up = ray(up);
}

episode::LocomotionCommand locomotion_input(
    const Camera3D& camera,
    const interaction::Pose& pose,
    bool frozen) {
    if (frozen) return {};
    vec3 forward(
        camera.target.x - camera.position.x,
        0.0F,
        camera.target.z - camera.position.z);
    if (length(forward) < 1.0e-5F) forward = vec3(0.0F, 0.0F, 1.0F);
    return episode::camera_relative_command(
        forward,
        pose.rotations[g1_skeleton::Simulation],
        {
            IsKeyDown(KEY_W),
            IsKeyDown(KEY_S),
            IsKeyDown(KEY_A),
            IsKeyDown(KEY_D),
        },
        0.22F);
}

bool edit_marker(interaction::Transform& marker, float dt) {
    bool changed = false;
    const float move = 0.55F * dt;
    if (IsKeyDown(KEY_LEFT)) {
        marker.position.x -= move;
        changed = true;
    }
    if (IsKeyDown(KEY_RIGHT)) {
        marker.position.x += move;
        changed = true;
    }
    if (IsKeyDown(KEY_UP)) {
        marker.position.z -= move;
        changed = true;
    }
    if (IsKeyDown(KEY_DOWN)) {
        marker.position.z += move;
        changed = true;
    }
    if (IsKeyDown(KEY_U)) {
        marker.position.y += move;
        changed = true;
    }
    if (IsKeyDown(KEY_J)) {
        marker.position.y -= move;
        changed = true;
    }
    if (IsKeyDown(KEY_Q) || IsKeyDown(KEY_E)) {
        const float sign = IsKeyDown(KEY_Q) ? 1.0F : -1.0F;
        marker.rotation = quat_normalize(quat_mul(
            quat_from_angle_axis(
                sign * 1.3F * dt, vec3(0.0F, 1.0F, 0.0F)),
            marker.rotation));
        changed = true;
    }
    return changed;
}

const char* state_name(episode::EpisodeState state, bool searching) {
    if (searching) return "SEARCHING";
    switch (state) {
        case episode::EpisodeState::FreeLocomotion: return "FREE";
        case episode::EpisodeState::Searching: return "SEARCHING";
        case episode::EpisodeState::Approach: return "APPROACH";
        case episode::EpisodeState::Bridge: return "BRIDGE";
        case episode::EpisodeState::Reach: return "REACH";
        case episode::EpisodeState::CarryBlend: return "CARRY BLEND";
        case episode::EpisodeState::Carry: return "CARRY";
        case episode::EpisodeState::PlaceApproach:
            return "PLACE APPROACH";
        case episode::EpisodeState::PlaceBridge: return "PLACE BRIDGE";
        case episode::EpisodeState::PlaceReach: return "PLACE REACH";
        case episode::EpisodeState::PlaceReturn: return "PLACE RETURN";
        case episode::EpisodeState::Failed: return "FAILED";
    }
    return "UNKNOWN";
}

void draw_oriented_box(
    const interaction::OrientedBox& box,
    Color color) {
    std::array<vec3, 8U> corners{};
    for (size_t corner = 0U; corner < corners.size(); ++corner) {
        const vec3 local(
            (corner & 1U ? 0.5F : -0.5F) * box.dimensions.x,
            (corner & 2U ? 0.5F : -0.5F) * box.dimensions.y,
            (corner & 4U ? 0.5F : -0.5F) * box.dimensions.z);
        corners[corner] = box.world.position +
            quat_mul_vec3(box.world.rotation, local);
    }
    for (size_t corner = 0U; corner < corners.size(); ++corner) {
        for (const size_t bit : {1U, 2U, 4U}) {
            const size_t other = corner ^ bit;
            if (corner < other) {
                DrawLine3D(ray(corners[corner]), ray(corners[other]), color);
            }
        }
    }
}

void draw_axes(
    const interaction::Transform& transform,
    float scale) {
    const std::array<vec3, 3U> axes = {
        vec3(scale, 0.0F, 0.0F),
        vec3(0.0F, scale, 0.0F),
        vec3(0.0F, 0.0F, scale),
    };
    const std::array<Color, 3U> colors = {RED, GREEN, BLUE};
    for (size_t axis = 0U; axis < axes.size(); ++axis) {
        DrawLine3D(
            ray(transform.position),
            ray(transform.position +
                quat_mul_vec3(transform.rotation, axes[axis])),
            colors[axis]);
    }
}

Color with_alpha(Color color, float alpha) {
    color.a = static_cast<unsigned char>(
        clampf(alpha, 0.0F, 1.0F) * 255.0F);
    return color;
}

void draw_bones(
    const interaction::WorldPose& pose,
    float alpha = 1.0F) {
    for (size_t bone = 0U; bone < g1_skeleton::BoneCount; ++bone) {
        const int32_t parent = g1_skeleton::kParents[bone];
        if (parent < 0) continue;
        DrawCylinderEx(
            ray(pose.positions[static_cast<size_t>(parent)]),
            ray(pose.positions[bone]),
            0.012F,
            0.012F,
            6,
            with_alpha(SKYBLUE, alpha));
    }
}

void draw_flat_bones(
    const episode::FlatSkeletonWorldPose& pose,
    float alpha = 1.0F) {
    for (size_t bone = 0U;
         bone < episode::kFlatSkeletonBoneCount;
         ++bone) {
        const int32_t parent = episode::kFlatSkeletonParents[bone];
        DrawSphere(
            ray(pose.positions[bone]),
            0.018F,
            with_alpha(DARKBLUE, alpha));
        if (parent < 0) continue;
        DrawCylinderEx(
            ray(pose.positions[static_cast<size_t>(parent)]),
            ray(pose.positions[bone]),
            0.012F,
            0.012F,
            6,
            with_alpha(SKYBLUE, alpha));
    }
}

void draw_plan(const episode::ReachPlan& plan, vec3 live_root) {
    vec3 previous_root = live_root;
    if (plan.entry_waypoints_world.empty()) {
        DrawLine3D(
            ray(previous_root),
            ray(plan.entry_root_world.position),
            Color{80, 80, 80, 180});
    } else {
        for (const vec3 waypoint : plan.entry_waypoints_world) {
            DrawLine3D(
                ray(previous_root),
                ray(waypoint),
                Color{80, 80, 80, 180});
            previous_root = waypoint;
        }
    }
    DrawSphere(ray(plan.entry_root_world.position), 0.045F, ORANGE);
    draw_axes(plan.entry_root_world, 0.18F);
    if (plan.reach.poses.size() < 2U) return;
    const size_t wrist = plan.hand == reach::Hand::Left
        ? g1_skeleton::LeftWrist
        : g1_skeleton::RightWrist;
    vec3 previous =
        interaction::world_pose(plan.reach.poses.front()).positions[wrist];
    for (size_t frame = 1U; frame < plan.reach.poses.size(); ++frame) {
        const vec3 current =
            interaction::world_pose(plan.reach.poses[frame]).positions[wrist];
        DrawLine3D(ray(previous), ray(current), LIME);
        previous = current;
    }
}

void draw_search_paths(
    const std::vector<std::vector<vec3>>& paths) {
    for (const std::vector<vec3>& path : paths) {
        if (path.size() < 2U) continue;
        vec3 previous = path.front();
        for (size_t sample = 3U; sample < path.size(); sample += 3U) {
            DrawLine3D(
                ray(previous),
                ray(path[sample]),
                Color{70, 150, 245, 105});
            previous = path[sample];
        }
        DrawLine3D(
            ray(previous), ray(path.back()), Color{70, 150, 245, 105});
    }
}

void draw_hud(
    const episode::InteractionEpisode& runtime,
    bool searching,
    size_t accepted,
    long long search_ms,
    const std::string& search_failure,
    bool show_mesh,
    bool show_bones) {
    DrawRectangle(14, 14, 760, 184, Color{255, 255, 255, 232});
    DrawText(
        "WASD move | arrows active marker | U/J height | Q/E yaw | F pick/place",
        26, 24, 17, DARKGRAY);
    DrawText(
        "Esc cancel | R reset | M mesh | B bones | mouse orbit/pan/zoom",
        26, 47, 16, DARKGRAY);
    const auto& output = runtime.output();
    const auto& attempt = runtime.attempt();
    const auto& place_attempt = runtime.place_attempt();
    const char* hand = output.selected_hand == interaction::Hand::Left
        ? "LEFT"
        : "RIGHT";
    const episode::ReachPlan* active_plan = place_attempt.has_value()
        ? &place_attempt->plan
        : attempt.has_value() ? &attempt->plan : nullptr;
    const size_t clip = active_plan != nullptr
        ? active_plan->reach.candidate.clip
        : 0U;
    DrawText(
        TextFormat(
            "STATE %s | HAND %s | CLIP %zu",
            state_name(runtime.state(), searching), hand, clip),
        26, 75, 21,
        runtime.state() == episode::EpisodeState::Failed ? MAROON : DARKBLUE);
    DrawText(
        TextFormat(
            "SEARCH %lld ms | ACCEPTED %zu | ATTACHED %s | PLACE %s",
            search_ms,
            accepted,
            output.attached ? "YES" : "NO",
            output.place_ready ? "READY" : "WAIT"),
        26, 103, 18, DARKGRAY);
    if (active_plan != nullptr) {
        DrawText(
            TextFormat(
                "ENTRY live %.2f m %.1f deg %.2f m/s | DIRECT %.3f | CONTACT %.2f cm %.1f deg",
                output.approach_distance_m,
                output.approach_yaw_error_radians *
                    180.0F / kPi,
                output.locomotion_speed_mps,
                active_plan->cost.directness_cost,
                active_plan->reach.position_error_m * 100.0F,
                active_plan->reach.orientation_error_radians *
                    180.0F / kPi),
            26, 129, 17, DARKGRAY);
    }
    const std::string failure = !output.failure.empty()
        ? output.failure
        : search_failure;
    DrawText(
        TextFormat(
            "MESH %s | BONES %s | FAILURE %s",
            show_mesh ? "ON" : "OFF",
            show_bones ? "ON" : "OFF",
            failure.empty() ? "none" : failure.c_str()),
        26, 157, 16, failure.empty() ? DARKGRAY : MAROON);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        const reach::Pack reach_pack = reach::load_pack(options.reach_pack);
        const std::filesystem::path walking_database =
            "resources/database.bin";
        episode::EpisodeConfig episode_config{};
        episode_config.attachment.required_hold_seconds = 0.50F;
        episode::InteractionEpisode runtime(
            walking_database,
            options.episode_pack / "carry_left_database.bin",
            options.episode_pack / "carry_right_database.bin",
            episode_config);
        const interaction::Transform table_world{
            vec3(0.0F, kTableTop - 0.5F * kTableDimensions.y, 0.0F),
            quat(),
        };
        const interaction::EnvironmentGeometry environment =
            interaction::make_coverage_environment(
                table_world, kTableDimensions);
        const auto supported_object =
            [&environment](size_t support_index) {
            const interaction::OrientedBox& support =
                environment.boxes.at(support_index);
            return interaction::Transform{
                support.world.position + vec3(
                    0.0F,
                    0.5F * support.dimensions.y +
                        0.5F * kObjectDimensions.y,
                    0.0F),
                quat(),
            };
        };
        std::optional<size_t> object_support_index{8U};
        std::optional<size_t> destination_support_index{5U};
        interaction::Transform object =
            supported_object(*object_support_index);
        interaction::Transform destination =
            supported_object(*destination_support_index);
        uint64_t object_generation = 1U;
        uint64_t destination_generation = 1U;
        uint64_t request_id = 1U;
        bool placement_synchronized = false;
        const episode::KnownGraspProvider grasp_provider(
            {
                vec3(0.5F * kObjectDimensions.x + 0.04F, 0.0F, 0.0F),
                quat_from_angle_axis(kPi, vec3(0.0F, 1.0F, 0.0F)),
            },
            vec3(-1.0F, 0.0F, 0.0F));
        reach::SearchConfig search_config{};
        search_config.deadline = std::chrono::seconds(30);
        search_config.worker_count = std::max<size_t>(
            1U,
            std::min<size_t>(
                8U,
                std::max(1U, std::thread::hardware_concurrency())));

        SetConfigFlags(FLAG_VSYNC_HINT | FLAG_MSAA_4X_HINT);
        InitWindow(1280, 800, "G1 motion-matched pickup episode");
        if (!IsWindowReady()) {
            throw std::runtime_error("episode viewer window failed to open");
        }
        SetExitKey(KEY_NULL);
        SetTargetFPS(60);
        Camera3D camera{};
        camera.fovy = 45.0F;
        camera.projection = CAMERA_PERSPECTIVE;
        OrbitCamera orbit{};
        update_camera(camera, orbit);

        G1MeshRenderer mesh{};
        std::array<char, 256U> mesh_error{};
        const bool mesh_loaded = ::g1_mesh_renderer_load(
            mesh,
            "resources/g1_mesh/g1_raylib.glb",
            mesh_error.data(),
            static_cast<int>(mesh_error.size()));
        bool show_mesh = false;
        bool show_bones = true;
        if (!mesh_loaded) {
            std::cerr << "G1 mesh unavailable: " << mesh_error.data()
                      << '\n';
        }
        std::optional<SearchJob> search;
        size_t accepted_count = 0U;
        long long search_ms = 0;
        std::string search_failure;
        std::vector<std::vector<vec3>> accepted_paths;

        while (!WindowShouldClose()) {
            const float dt = std::min(GetFrameTime(), 0.10F);
            update_camera(camera, orbit);
            if (IsKeyPressed(KEY_M) && mesh.loaded) show_mesh = !show_mesh;
            if (IsKeyPressed(KEY_B)) show_bones = !show_bones;

            const bool pickup_editable =
                runtime.state() == episode::EpisodeState::FreeLocomotion &&
                !search.has_value();
            const bool destination_editable =
                runtime.state() == episode::EpisodeState::Carry &&
                runtime.output().place_ready &&
                !search.has_value();
            const bool object_changed =
                pickup_editable && edit_marker(object, dt);
            const bool destination_changed =
                destination_editable && edit_marker(destination, dt);
            if (object_changed) {
                object_support_index = episode::find_support_index(
                    environment, object, kObjectDimensions);
                ++object_generation;
                accepted_paths.clear();
            }
            if (destination_changed) {
                destination_support_index = episode::find_support_index(
                    environment, destination, kObjectDimensions);
                ++destination_generation;
                accepted_paths.clear();
            }

            if (IsKeyPressed(KEY_R)) {
                if (search.has_value()) {
                    search->cancellation->store(
                        true, std::memory_order_relaxed);
                    search->result.wait();
                    search.reset();
                }
                runtime.reset();
                object_support_index = 8U;
                destination_support_index = 5U;
                object = supported_object(*object_support_index);
                destination =
                    supported_object(*destination_support_index);
                ++object_generation;
                ++destination_generation;
                placement_synchronized = false;
                accepted_count = 0U;
                search_ms = 0;
                search_failure.clear();
                accepted_paths.clear();
            }

            if (IsKeyPressed(KEY_F) &&
                (pickup_editable || destination_editable)) {
                const SearchPurpose purpose = destination_editable
                    ? SearchPurpose::Place
                    : SearchPurpose::Pickup;
                if (purpose == SearchPurpose::Place &&
                    !destination_support_index.has_value()) {
                    search_failure =
                        "destination is not on a horizontal support";
                    continue;
                }
                episode::ObjectSnapshot snapshot{
                    purpose == SearchPurpose::Pickup
                        ? object_generation
                        : destination_generation,
                    purpose == SearchPurpose::Pickup
                        ? object
                        : destination,
                    kObjectDimensions,
                };
                const std::vector<episode::GraspCandidate> grasps =
                    grasp_provider.query(snapshot);
                if (!grasps.empty()) {
                    SearchJob job{};
                    job.purpose = purpose;
                    job.object = snapshot;
                    job.grasp = grasps.front();
                    job.query = {
                        job.grasp.hand_world,
                        job.grasp.approach_world,
                    };
                    job.object_box = {snapshot.world, snapshot.dimensions};
                    job.environment = environment;
                    if (purpose == SearchPurpose::Place) {
                        const interaction::OrientedBox& support =
                            environment.boxes.at(
                                *destination_support_index);
                        job.support = {
                            support.world,
                            support.dimensions,
                        };
                        job.required_hand =
                            runtime.output().selected_hand ==
                                    interaction::Hand::Left
                                ? reach::Hand::Left
                                : reach::Hand::Right;
                    }
                    const reach::ExhaustiveQuery query = job.query;
                    const interaction::OrientedBox box = job.object_box;
                    const interaction::EnvironmentGeometry geometry =
                        job.environment;
                    job.cancellation =
                        std::make_shared<std::atomic_bool>(false);
                    reach::SearchConfig request_config = search_config;
                    request_config.cancellation = job.cancellation;
                    job.result = std::async(
                        std::launch::async,
                        [&reach_pack, query, box, geometry, request_config] {
                            return reach::search_all(
                                reach_pack,
                                query,
                                box,
                                geometry,
                                request_config);
                        });
                    search.emplace(std::move(job));
                    search_failure.clear();
                }
            }

            if (search.has_value() &&
                search->result.wait_for(std::chrono::seconds(0)) ==
                    std::future_status::ready) {
                reach::SearchResult result = search->result.get();
                search_ms = std::chrono::duration_cast<
                    std::chrono::milliseconds>(result.elapsed).count();
                accepted_count = 0U;
                accepted_paths.clear();
                accepted_paths.reserve(result.accepted.size());
                for (const size_t index : result.accepted) {
                    const reach::Evaluation& evaluation =
                        result.evaluations.at(index).evaluation;
                    const reach::Hand hand = static_cast<reach::Hand>(
                        reach_pack.database.active_hands.at(
                            evaluation.candidate.clip));
                    if (search->required_hand.has_value() &&
                        hand != *search->required_hand) {
                        continue;
                    }
                    ++accepted_count;
                    accepted_paths.push_back(
                        result.evaluations.at(index).hand_path);
                }
                std::array<size_t, reach::kRejectionCount> rejected{};
                for (const reach::CompactEvaluation& value :
                     result.evaluations) {
                    ++rejected.at(static_cast<size_t>(
                        value.evaluation.rejection));
                }
                std::cerr
                    << "episode search processed " << result.processed
                    << "/" << result.total
                    << " accepted " << accepted_count
                    << " rejection histogram";
                for (size_t count : rejected) std::cerr << " " << count;
                std::cerr << '\n';
                const std::optional<episode::ReachPlan> plan =
                    episode::choose_reach_plan(
                        reach_pack,
                        result,
                        search->query,
                        search->object_box,
                        search->environment,
                        search_config,
                        runtime.output().pose,
                        search->grasp,
                        search->required_hand);
                if (plan.has_value()) {
                    const vec3 live_root = runtime.output().pose.positions[
                        g1_skeleton::Simulation];
                    const vec3 entry = plan->entry_root_world.position;
                    std::cerr
                        << "episode plan clip "
                        << plan->reach.candidate.clip
                        << " live root " << live_root.x << " "
                        << live_root.y << " " << live_root.z
                        << " entry " << entry.x << " "
                        << entry.y << " " << entry.z << '\n';
                    const uint64_t request = request_id++;
                    const bool committed =
                        search->purpose == SearchPurpose::Pickup
                        ? runtime.commit({
                              search->object,
                              search->grasp,
                              *plan,
                              request,
                          })
                        : runtime.commit_place({
                              search->object,
                              search->grasp,
                              *plan,
                              search->support,
                              request,
                          });
                    if (!committed) {
                        search_failure =
                            search->purpose == SearchPurpose::Pickup
                            ? "pickup plan commit rejected"
                            : "place plan commit rejected";
                    }
                } else {
                    search_failure = result.complete
                        ? "no playable collision-free entry"
                        : "search deadline exceeded";
                }
                search.reset();
            }

            const bool escape = IsKeyPressed(KEY_ESCAPE);
            const episode::LocomotionCommand command = locomotion_input(
                camera,
                runtime.output().pose,
                search.has_value());
            runtime.update({
                search.has_value() ? 0.0F : dt,
                command,
                object_generation,
                escape,
                false,
                destination_generation,
            });
            if (runtime.output().placed && !placement_synchronized) {
                const interaction::Transform previous_object = object;
                object = runtime.output().object_world;
                destination = previous_object;
                std::swap(
                    object_support_index,
                    destination_support_index);
                ++object_generation;
                ++destination_generation;
                placement_synchronized = true;
                accepted_paths.clear();
            } else if (!runtime.output().placed) {
                placement_synchronized = false;
            }
            const interaction::Transform displayed_object =
                runtime.output().attached
                ? runtime.output().object_world
                : object;
            interaction::WorldPose world =
                interaction::world_pose(runtime.output().pose);
            const bool flat_visible =
                runtime.output().flat_locomotion.valid;
            if (show_mesh && mesh.loaded && !flat_visible &&
                !::g1_mesh_renderer_update(
                    mesh,
                    slice1d<vec3>(
                        static_cast<int>(g1_skeleton::BoneCount),
                        world.positions.data()),
                    slice1d<quat>(
                        static_cast<int>(g1_skeleton::BoneCount),
                        world.rotations.data()),
                    mesh_error.data(),
                    static_cast<int>(mesh_error.size()))) {
                show_mesh = false;
                show_bones = true;
            }

            BeginDrawing();
            ClearBackground(Color{238, 241, 245, 255});
            BeginMode3D(camera);
            DrawGrid(30, 0.20F);
            for (const interaction::OrientedBox& box : environment.boxes) {
                DrawCubeV(
                    ray(box.world.position),
                    ray(box.dimensions),
                    Color{136, 104, 76, 175});
                draw_oriented_box(box, Color{75, 54, 38, 255});
            }
            const interaction::OrientedBox displayed_box{
                displayed_object, kObjectDimensions};
            DrawCubeV(
                ray(displayed_object.position),
                ray(kObjectDimensions),
                Color{245, 176, 35, 220});
            draw_oriented_box(displayed_box, ORANGE);
            const interaction::OrientedBox destination_box{
                destination, kObjectDimensions};
            DrawCubeV(
                ray(destination.position),
                ray(kObjectDimensions),
                Color{75, 205, 115, 65});
            draw_oriented_box(destination_box, DARKGREEN);
            const auto grasp = grasp_provider.query({
                object_generation, displayed_object, kObjectDimensions});
            if (!grasp.empty()) draw_axes(grasp.front().hand_world, 0.14F);
            if (runtime.place_attempt().has_value()) {
                draw_plan(
                    runtime.place_attempt()->plan,
                    runtime.output().pose.positions[
                        g1_skeleton::Simulation]);
            } else if (runtime.attempt().has_value()) {
                draw_plan(
                    runtime.attempt()->plan,
                    runtime.output().pose.positions[
                        g1_skeleton::Simulation]);
            }
            draw_search_paths(accepted_paths);
            if (flat_visible) {
                const float bridge_alpha =
                    runtime.output().bridge_alpha;
                draw_flat_bones(
                    runtime.output().flat_locomotion,
                    1.0F - bridge_alpha);
                if (runtime.state() == episode::EpisodeState::Bridge &&
                    show_bones) {
                    draw_bones(world, bridge_alpha);
                }
            } else {
                if (show_mesh && mesh.loaded) {
                    ::g1_mesh_renderer_draw(mesh);
                }
                if (show_bones) draw_bones(world);
            }
            EndMode3D();
            draw_hud(
                runtime,
                search.has_value(),
                accepted_count,
                search_ms,
                search_failure,
                show_mesh,
                show_bones);
            EndDrawing();
        }

        if (search.has_value()) {
            search->cancellation->store(
                true, std::memory_order_relaxed);
            search->result.wait();
        }
        ::g1_mesh_renderer_unload(mesh);
        CloseWindow();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "g1 interaction episode viewer FAILED: "
                  << error.what() << '\n';
        return 1;
    }
}
