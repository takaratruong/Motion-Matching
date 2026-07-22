#include "interaction_hand_trajectories.h"
#include "interaction_pose.h"
#include "raylib.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using interaction::HandTrajectory;
using interaction::HandTrajectoryQuery;
using interaction::OrientedBox;
using interaction::ShelfGeometry;
using interaction::ShapedHandTrajectory;
using interaction::TrajectoryFeasibilityReason;

struct RenderedTrajectory {
    HandTrajectory source;
    ShapedHandTrajectory shaped;
    TrajectoryFeasibilityReason reason = TrajectoryFeasibilityReason::None;
};

struct TrajectorySet {
    std::vector<RenderedTrajectory> valid;
    std::vector<RenderedTrajectory> rejected;
    size_t compatible = 0U;
    size_t ik_rejected = 0U;
    size_t object_rejected = 0U;
    size_t table_rejected = 0U;
};

Vector3 ray_vector(vec3 value) {
    return Vector3{value.x, value.y, value.z};
}

vec3 read_vec3(const std::vector<float>& values, size_t index) {
    const size_t offset = index * 3U;
    return vec3(values.at(offset), values.at(offset + 1U), values.at(offset + 2U));
}

quat read_quat(const std::vector<float>& values, size_t index) {
    const size_t offset = index * 4U;
    return quat_normalize(quat(
        values.at(offset), values.at(offset + 1U),
        values.at(offset + 2U), values.at(offset + 3U)));
}

struct CanonicalGrasp {
    size_t clip = 0U;
    int32_t contact_frame = -1;
    interaction::Hand hand = interaction::Hand::Right;
    vec3 dimensions{};
    interaction::Transform grasp_in_object{};
    interaction::Transform source_object{};
    interaction::Transform table_world{};
    vec3 table_dimensions{};
};

CanonicalGrasp find_canonical_grasp(const interaction::Database& database) {
    constexpr uint8_t contact_phase = 2U;
    for (size_t clip = 0U; clip < database.clip_count; ++clip) {
        const int32_t start = database.range_starts.at(clip);
        const int32_t stop = database.range_stops.at(clip);
        for (int32_t frame = start; frame < stop; ++frame) {
            if (database.phases.at(static_cast<size_t>(frame)) != contact_phase) {
                continue;
            }
            const int32_t object_frame = std::max(start, frame - 1);
            return {
                clip,
                frame,
                static_cast<interaction::Hand>(database.active_hands.at(clip)),
                read_vec3(database.object_dimensions, clip),
                {
                    read_vec3(database.grasp_positions_object, clip),
                    read_quat(database.grasp_rotations_object, clip),
                },
                {
                    read_vec3(
                        database.object_positions,
                        static_cast<size_t>(object_frame)),
                    read_quat(
                        database.object_rotations,
                        static_cast<size_t>(object_frame)),
                },
                {
                    read_vec3(database.table_positions, clip),
                    read_quat(database.table_rotations, clip),
                },
                read_vec3(database.table_sizes, clip),
            };
        }
    }
    throw std::runtime_error("interaction database has no Contact clip");
}

HandTrajectoryQuery make_query(
    const interaction::Transform& object_world,
    const CanonicalGrasp& canonical,
    bool position_only) {
    const interaction::Transform grasp_world = interaction::compose(
        object_world, canonical.grasp_in_object);
    HandTrajectoryQuery query{};
    query.object_world = object_world;
    query.object_dimensions = canonical.dimensions;
    query.hand = canonical.hand;
    query.grasp_world_position = grasp_world.position;
    query.grasp_world_rotation = grasp_world.rotation;
    query.constrain_grasp_orientation = !position_only;
    return query;
}

TrajectorySet rebuild_valid_trajectories(
    const interaction::Database& database,
    const HandTrajectoryQuery& query,
    const ShelfGeometry& shelf) {
    const OrientedBox object{query.object_world, query.object_dimensions};
    const std::vector<HandTrajectory> candidates =
        interaction::select_hand_trajectories(database, query);
    TrajectorySet result{};
    result.compatible = candidates.size();
    result.valid.reserve(candidates.size());
    result.rejected.reserve(candidates.size());
    for (const HandTrajectory& candidate : candidates) {
        ShapedHandTrajectory shaped = interaction::shape_hand_trajectory(
            database, candidate, query);
        if (!shaped.contact_accepted) {
            ++result.ik_rejected;
            result.rejected.push_back({
                candidate, std::move(shaped),
                TrajectoryFeasibilityReason::None});
            continue;
        }
        const auto feasibility =
            interaction::evaluate_shaped_trajectory_feasibility(
                shaped, candidate.contact_point, query.hand, object, shelf);
        RenderedTrajectory rendered{
            candidate, std::move(shaped), feasibility.reason};
        if (feasibility.reason == TrajectoryFeasibilityReason::None) {
            result.valid.push_back(std::move(rendered));
        } else {
            if (feasibility.reason ==
                TrajectoryFeasibilityReason::ObjectCollision) {
                ++result.object_rejected;
            } else {
                ++result.table_rejected;
            }
            result.rejected.push_back(std::move(rendered));
        }
    }
    return result;
}

void draw_oriented_box(const OrientedBox& box, Color color) {
    const vec3 half = box.dimensions * 0.5F;
    const std::array<vec3, 8> local = {
        vec3(-half.x, -half.y, -half.z), vec3(half.x, -half.y, -half.z),
        vec3(half.x, half.y, -half.z), vec3(-half.x, half.y, -half.z),
        vec3(-half.x, -half.y, half.z), vec3(half.x, -half.y, half.z),
        vec3(half.x, half.y, half.z), vec3(-half.x, half.y, half.z),
    };
    std::array<vec3, 8> world{};
    for (size_t corner = 0U; corner < local.size(); ++corner) {
        world[corner] = interaction::compose(
            box.world, interaction::Transform{local[corner], quat()}).position;
    }
    constexpr std::array<std::array<size_t, 2>, 12> edges = {{
        {{0U, 1U}}, {{1U, 2U}}, {{2U, 3U}}, {{3U, 0U}},
        {{4U, 5U}}, {{5U, 6U}}, {{6U, 7U}}, {{7U, 4U}},
        {{0U, 4U}}, {{1U, 5U}}, {{2U, 6U}}, {{3U, 7U}},
    }};
    for (const auto& edge : edges) {
        DrawLine3D(ray_vector(world[edge[0]]), ray_vector(world[edge[1]]), color);
    }
}

void draw_path(
    const RenderedTrajectory& trajectory,
    Color color,
    bool emphasized = false) {
    const auto& hands = trajectory.shaped.path.hands;
    for (size_t sample = 1U; sample < hands.size(); ++sample) {
        const Vector3 start = ray_vector(
            hands[sample - 1U].position);
        const Vector3 stop = ray_vector(
            hands[sample].position);
        if (emphasized) {
            DrawCylinderEx(start, stop, 0.009F, 0.009F, 6, color);
            DrawSphere(stop, 0.014F, color);
        } else {
            DrawLine3D(start, stop, color);
        }
    }
    if (!hands.empty()) {
        DrawSphere(
            ray_vector(hands[trajectory.source.contact_point].position),
            0.018F, color);
    }
}

const char* phase_name(uint8_t phase) {
    switch (phase) {
        case 1U: return "REACH";
        case 2U: return "CONTACT";
        case 3U: return "LIFT";
        default: return "TRANSITION";
    }
}

size_t animated_sample(
    const RenderedTrajectory& trajectory,
    float animation_seconds) {
    if (trajectory.shaped.poses.empty()) {
        throw std::invalid_argument("selected trajectory has no shaped poses");
    }
    return static_cast<size_t>(std::floor(animation_seconds * 25.0F)) %
        trajectory.shaped.poses.size();
}

void draw_selected_skeleton(
    const interaction::Pose& shaped_pose,
    Color joint_color,
    Color bone_color) {
    const interaction::WorldPose shaped = interaction::world_pose(shaped_pose);
    for (size_t bone = 0U; bone < shaped.positions.size(); ++bone) {
        DrawSphereWires(
            ray_vector(shaped.positions[bone]), 0.024F, 4, 8, joint_color);
        const int32_t parent = g1_skeleton::kParents[bone];
        if (parent >= 0) {
            DrawCylinderEx(
                ray_vector(shaped.positions[static_cast<size_t>(parent)]),
                ray_vector(shaped.positions[bone]),
                0.014F, 0.014F, 6, bone_color);
        }
    }
}

void apply_rotation(
    interaction::Transform& object_world,
    float angle,
    vec3 axis) {
    object_world.rotation = quat_normalize(quat_mul(
        quat_from_angle_axis(angle, axis), object_world.rotation));
}

std::filesystem::path pack_path(int argc, char** argv) {
    if (argc > 2) throw std::invalid_argument("usage: hand_trajectory_viewer [pack]");
    if (argc == 2) return argv[1];
    const char* configured = std::getenv("MM_INTERACTION_PACK");
    return configured != nullptr
        ? std::filesystem::path(configured)
        : std::filesystem::path("build/smart-pickup/full-pack");
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::filesystem::path pack = pack_path(argc, argv);
        const interaction::Database database = interaction::load_database(
            pack / "interaction_database.bin");
        const CanonicalGrasp canonical = find_canonical_grasp(database);
        const vec3 scene_offset(
            -canonical.table_world.position.x,
            0.0F,
            -canonical.table_world.position.z);
        interaction::Transform table_world = canonical.table_world;
        table_world.position = table_world.position + scene_offset;
        const ShelfGeometry table_geometry =
            interaction::make_recorded_table_geometry(
                table_world, canonical.table_dimensions);

        interaction::Transform initial_object = canonical.source_object;
        initial_object.position = initial_object.position + scene_offset;
        interaction::Transform object_world = initial_object;
        bool position_only = false;
        bool show_rejected = true;
        size_t selected_index = 0U;
        float animation_seconds = 0.0F;
        HandTrajectoryQuery query = make_query(
            object_world, canonical, position_only);
        TrajectorySet trajectories = rebuild_valid_trajectories(
            database, query, table_geometry);

        SetConfigFlags(FLAG_VSYNC_HINT | FLAG_MSAA_4X_HINT);
        InitWindow(1280, 800, "Generic grasp kinematic trajectory lab");
        if (!IsWindowReady()) {
            throw std::runtime_error("trajectory viewer could not open a window");
        }
        SetTargetFPS(60);
        Camera3D camera{};
        camera.position = Vector3{2.35F, 1.55F, -2.45F};
        camera.target = Vector3{0.0F, 0.70F, 0.0F};
        camera.up = Vector3{0.0F, 1.0F, 0.0F};
        camera.fovy = 45.0F;
        camera.projection = CAMERA_PERSPECTIVE;

        while (!WindowShouldClose()) {
            const float dt = std::min(GetFrameTime(), 0.05F);
            animation_seconds += dt;
            const float translation_step = 0.45F * dt;
            const float rotation_step = 0.9F * dt;
            bool object_changed = false;
            if (IsKeyDown(KEY_LEFT)) {
                object_world.position.x -= translation_step;
                object_changed = true;
            }
            if (IsKeyDown(KEY_RIGHT)) {
                object_world.position.x += translation_step;
                object_changed = true;
            }
            if (IsKeyDown(KEY_UP)) {
                object_world.position.z += translation_step;
                object_changed = true;
            }
            if (IsKeyDown(KEY_DOWN)) {
                object_world.position.z -= translation_step;
                object_changed = true;
            }
            if (IsKeyDown(KEY_W) || IsKeyDown(KEY_PAGE_UP)) {
                object_world.position.y += translation_step;
                object_changed = true;
            }
            if (IsKeyDown(KEY_S) || IsKeyDown(KEY_PAGE_DOWN)) {
                object_world.position.y -= translation_step;
                object_changed = true;
            }
            if (IsKeyDown(KEY_Q)) {
                apply_rotation(object_world, rotation_step, vec3(0.0F, 1.0F, 0.0F));
                object_changed = true;
            }
            if (IsKeyDown(KEY_E)) {
                apply_rotation(object_world, -rotation_step, vec3(0.0F, 1.0F, 0.0F));
                object_changed = true;
            }
            if (IsKeyDown(KEY_R)) {
                apply_rotation(object_world, rotation_step, vec3(1.0F, 0.0F, 0.0F));
                object_changed = true;
            }
            if (IsKeyDown(KEY_F)) {
                apply_rotation(object_world, -rotation_step, vec3(1.0F, 0.0F, 0.0F));
                object_changed = true;
            }
            if (IsKeyDown(KEY_Z)) {
                apply_rotation(object_world, rotation_step, vec3(0.0F, 0.0F, 1.0F));
                object_changed = true;
            }
            if (IsKeyDown(KEY_C)) {
                apply_rotation(object_world, -rotation_step, vec3(0.0F, 0.0F, 1.0F));
                object_changed = true;
            }
            if (IsKeyPressed(KEY_BACKSPACE)) {
                object_world = initial_object;
                object_changed = true;
            }
            if (IsKeyPressed(KEY_LEFT_BRACKET)) {
                if (!trajectories.valid.empty()) {
                    selected_index = selected_index == 0U
                        ? trajectories.valid.size() - 1U
                        : selected_index - 1U;
                    animation_seconds = 0.0F;
                }
            }
            if (IsKeyPressed(KEY_SLASH) ||
                IsKeyPressed(KEY_RIGHT_BRACKET)) {
                if (!trajectories.valid.empty()) {
                    selected_index =
                        (selected_index + 1U) % trajectories.valid.size();
                    animation_seconds = 0.0F;
                }
            }
            if (IsKeyPressed(KEY_V)) show_rejected = !show_rejected;
            if (IsKeyPressed(KEY_P)) {
                position_only = !position_only;
                object_changed = true;
            }
            if (IsKeyPressed(KEY_ENTER)) object_changed = true;
            if (object_changed) {
                query = make_query(object_world, canonical, position_only);
                trajectories = rebuild_valid_trajectories(
                    database, query, table_geometry);
                selected_index = 0U;
            }

            BeginDrawing();
            ClearBackground(Color{238, 241, 245, 255});
            BeginMode3D(camera);
            DrawGrid(20, 0.25F);
            for (const OrientedBox& box : table_geometry.boxes) {
                DrawCubeV(
                    ray_vector(box.world.position), ray_vector(box.dimensions),
                    Color{135, 102, 74, 155});
                draw_oriented_box(box, Color{72, 52, 39, 255});
            }
            draw_oriented_box(
                {object_world, canonical.dimensions},
                Color{255, 177, 35, 255});
            DrawSphere(ray_vector(query.grasp_world_position), 0.028F, GOLD);
            if (show_rejected) {
                for (const RenderedTrajectory& rejected :
                     trajectories.rejected) {
                    draw_path(rejected, Color{210, 45, 55, 75});
                }
            }
            for (size_t index = 0U; index < trajectories.valid.size(); ++index) {
                if (index == selected_index) continue;
                const float fraction = trajectories.valid.size() <= 1U
                    ? 0.0F
                    : static_cast<float>(index) /
                      static_cast<float>(trajectories.valid.size() - 1U);
                draw_path(
                    trajectories.valid[index],
                    ColorFromHSV(125.0F + 95.0F * fraction, 0.78F, 0.86F));
            }
            if (!trajectories.valid.empty()) {
                const RenderedTrajectory& selected =
                    trajectories.valid[selected_index];
                const size_t sample = animated_sample(
                    selected, animation_seconds);
                draw_path(selected, LIME, true);
                draw_selected_skeleton(
                    selected.shaped.poses[sample], DARKBLUE, SKYBLUE);
            }
            EndMode3D();

            DrawRectangle(14, 14, 665, 168, Color{255, 255, 255, 225});
            DrawText("Generic grasp trajectory field", 26, 24, 24, DARKGRAY);
            DrawText(
                TextFormat(
                    "compatible %i | valid %i | IK-rejected %i | object %i | table %i",
                    static_cast<int>(trajectories.compatible),
                    static_cast<int>(trajectories.valid.size()),
                    static_cast<int>(trajectories.ik_rejected),
                    static_cast<int>(trajectories.object_rejected),
                    static_cast<int>(trajectories.table_rejected)),
                26, 56, 18, DARKGRAY);
            if (trajectories.valid.empty()) {
                DrawText(
                    "0 valid motions for this world grasp",
                    26, 82, 16, MAROON);
            } else {
                const RenderedTrajectory& selected =
                    trajectories.valid[selected_index];
                const size_t sample = animated_sample(
                    selected, animation_seconds);
                const int32_t frame = selected.source.reach_frame +
                    static_cast<int32_t>(sample);
                DrawText(
                    TextFormat(
                        "option %i/%i | clip %i | SAFE | %s frame %i",
                        static_cast<int>(selected_index + 1U),
                        static_cast<int>(trajectories.valid.size()),
                        selected.source.clip,
                        phase_name(database.phases.at(
                            static_cast<size_t>(frame))),
                        frame),
                    26, 82, 16, DARKGRAY);
            }
            DrawText(
                "Arrows: X/Z  W/S: height  Q/E: yaw  R/F: pitch  Z/C: roll",
                26, 108, 16, DARKGRAY);
            DrawText(
                TextFormat(
                    "[: previous  / or ]: next  Enter: rerun  P: %s",
                    position_only ? "position-only grasp" : "full grasp pose"),
                26, 134, 16, DARKGRAY);
            EndDrawing();
        }
        CloseWindow();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "hand trajectory viewer: " << error.what() << '\n';
        return 2;
    }
}
